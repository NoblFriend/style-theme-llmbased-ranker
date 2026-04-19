from contextlib import asynccontextmanager
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
import re
import yaml
import uuid
import json
import math
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

# --- Configuration ---
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

MODEL_ID = config["model"]["name"]
# Use device from config, fallback to auto-detection if not specified or invalid
# Supports: "cuda", "cuda:0", "cuda:0,1,2", [0, 1, 2], ["cuda:0", "cuda:1"]
device_config = config["model"].get("device", "cuda")
print(f"[INFO] Device config: {device_config}")

def parse_device_config(device_cfg):
    """
    Parse device configuration and return (cuda_visible_devices, tensor_parallel_size).
    
    Supports formats:
    - "cuda" -> use all available GPUs with TP=1
    - "cuda:0" -> single GPU
    - "cuda:0,1,2" or "0,1,2" -> multiple GPUs
    - [0, 1, 2] or ["cuda:0", "cuda:1"] -> list of GPUs
    """
    if device_cfg is None or device_cfg == "cuda":
        return None, 1  # Use default, single GPU
    
    gpu_ids = []
    
    if isinstance(device_cfg, list):
        # List format: [0, 1, 2] or ["cuda:0", "cuda:1"]
        for item in device_cfg:
            if isinstance(item, int):
                gpu_ids.append(str(item))
            elif isinstance(item, str):
                # Extract number from "cuda:N" or just "N"
                if "cuda:" in item:
                    gpu_ids.append(item.split(":")[1])
                else:
                    gpu_ids.append(item.strip())
    elif isinstance(device_cfg, str):
        # String format: "cuda:0", "cuda:0,1,2", "0,1,2"
        if "cuda:" in device_cfg:
            # "cuda:0" or "cuda:0,1,2"
            ids_part = device_cfg.split("cuda:")[-1]
            gpu_ids = [x.strip() for x in ids_part.split(",")]
        elif "," in device_cfg:
            # "0,1,2"
            gpu_ids = [x.strip() for x in device_cfg.split(",")]
        else:
            # Just a number "0"
            gpu_ids = [device_cfg.strip()]
    elif isinstance(device_cfg, int):
        gpu_ids = [str(device_cfg)]
    
    if not gpu_ids:
        return None, 1
    
    cuda_visible = ",".join(gpu_ids)
    tensor_parallel = len(gpu_ids)
    
    return cuda_visible, tensor_parallel

CUDA_VISIBLE_DEVICES, TENSOR_PARALLEL_SIZE = parse_device_config(device_config)
print(f"[INFO] CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}, tensor_parallel_size={TENSOR_PARALLEL_SIZE}")

# --- Global State ---
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load Model on Startup
    print(f"Loading model: {MODEL_ID}...")
    print(f"[INFO] Tensor Parallel Size: {TENSOR_PARALLEL_SIZE}")
    try:
        # Set CUDA_VISIBLE_DEVICES if specified
        if CUDA_VISIBLE_DEVICES is not None:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
            print(f"[INFO] Set CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}")

        # Load tokenizer for chat template application
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        # Calculate available GPU memory to avoid OOM if other processes are running
        gpu_memory_utilization = 0.90
        try:
            import torch
            if torch.cuda.is_available():
                # Get memory from first visible device
                free_mem, total_mem = torch.cuda.mem_get_info(0)
                
                # We want to use a safe fraction of the *currently free* memory.
                # vLLM's gpu_memory_utilization is a fraction of the *total* memory.
                # So we calculate: (Free Memory * 0.9) / Total Memory
                # Using 0.9 of free memory leaves a 10% buffer of the free space.
                
                safe_utilization = (free_mem * 0.9) / total_mem
                
                # Clamp values
                if safe_utilization > 0.9: safe_utilization = 0.9
                if safe_utilization < 0.1: safe_utilization = 0.1
                
                gpu_memory_utilization = safe_utilization
                print(f"[INFO] GPU 0 Memory: Free={free_mem/1024**3:.2f}GB, Total={total_mem/1024**3:.2f}GB")
                print(f"[INFO] Auto-adjusting gpu_memory_utilization to {gpu_memory_utilization:.4f}")
        except Exception as e:
            print(f"[WARN] Failed to detect GPU memory, using default utilization: {e}")

        # Initialize vLLM engine with tensor parallelism
        engine_args = AsyncEngineArgs(
            model=MODEL_ID,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            max_model_len=config["model"].get("max_context_length", 32768),
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True, # Often needed for some models
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        ml_models["tokenizer"] = tokenizer
        ml_models["engine"] = engine
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
    
    yield
    
    # 2. Clean up on Shutdown
    print("Shutting down...")
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

# --- Data Models ---
class RankRequest(BaseModel):
    style_name: str
    topic: Optional[str] = None
    texts: List[str]

class RankResponse(BaseModel):
    ranked_indices: List[int]
    ranked_texts: List[str]
    raw_output: Optional[str] = None

# New scoring models
class ScoreRequest(BaseModel):
    style_name: str
    topic: str  # Required for scoring
    text: str   # Single text to score

class CriteriaScores(BaseModel):
    topic_relevant: bool  # Hard gate: is text about the topic at all?
    style_match: float    # 0-1: how well text matches the style
    topic_depth: float    # 0-1: depth of topic coverage
    topic_style_coherence: float  # 0-1: how naturally style integrates with topic

class ScoreResponse(BaseModel):
    final_score: float
    criteria: CriteriaScores
    raw_output: Optional[str] = None

# --- Scoring Functions ---
def compute_final_score(scores: Dict[str, float], topic_relevant: bool) -> float:
    """
    Compute final score from criteria scores.
    Uses a blend of arithmetic and geometric mean for balance.
    """
    if not topic_relevant:
        return 0.0
    
    style = scores["style_match"]
    depth = scores["topic_depth"]
    coherence = scores["topic_style_coherence"]
    
    # Weighted arithmetic mean
    weights = {
        "style_match": 0.25,
        "topic_depth": 0.25,
        "topic_style_coherence": 0.50,  # Most important - integration
    }
    arithmetic_mean = (
        style * weights["style_match"] +
        depth * weights["topic_depth"] +
        coherence * weights["topic_style_coherence"]
    )
    
    # Geometric mean (penalizes low outliers)
    geometric_mean = (style * depth * coherence) ** (1/3)
    
    # Blend: 60% arithmetic, 40% geometric
    final = 0.6 * arithmetic_mean + 0.4 * geometric_mean
    
    return round(max(0.0, min(1.0, final)), 3)

# --- Helper ---
def format_prompt(style_name: str, texts: List[str], topic: Optional[str] = None):
    passages_text = ""
    for idx, text in enumerate(texts):
        # Using 1-based indexing for the prompt
        passages_text += f"[{idx+1}] {text}\n"
    
    num = len(texts)
    
    if topic:
        user_content = f"""I will provide you with {num} texts, each indicated by a number identifier [].
Rank the texts based on how well they match the style: "{style_name}" AND the topic: "{topic}".

Ranking Criteria:
1. Best: The text clearly matches BOTH the style "{style_name}" AND the topic "{topic}".
2. Medium: The text matches EITHER the style "{style_name}" OR the topic "{topic}", but not both. (e.g. correct style but wrong topic, or correct topic but wrong style). These should be ranked similarly.
3. Worst: The text matches NEITHER the style nor the topic.

Texts:
{passages_text}

Ranking Task: Rank the {num} texts above based on the criteria.
The texts should be listed in descending order (best match first) using their identifiers.
The output format must be strictly: [best_id] > [second_best_id] > ...
Example: [1] > [3] > [2]
Do not provide any explanation, only the ranking.
"""
    else:
        # Listwise ranking prompt
        user_content = f"""I will provide you with {num} texts, each indicated by a number identifier [].
Rank the texts based on how well they match the style: "{style_name}".

Texts:
{passages_text}

Ranking Task: Rank the {num} texts above based on their adherence to the style "{style_name}".
The texts should be listed in descending order (best match first) using their identifiers.
The output format must be strictly: [best_id] > [second_best_id] > ...
Example: [1] > [3] > [2]
Do not provide any explanation, only the ranking.
"""

    messages = [
        {"role": "system", "content": "You are an expert literary critic and style analyzer. Your task is to rank texts based on specific stylistic criteria."},
        {"role": "user", "content": user_content}
    ]
    return messages

def format_score_prompt(style_name: str, topic: str, text: str) -> List[dict]:
    """
    Format prompt for pointwise scoring of a single text.
    Returns structured JSON with multiple criteria scores.
    """
    user_content = f"""Evaluate this text on multiple criteria. 
Topic required: "{topic}"
Style required: "{style_name}"

TEXT TO EVALUATE:
\"\"\"{text}\"\"\"

EVALUATION CRITERIA:

1. topic_relevant (true/false): Is this text ACTUALLY ABOUT "{topic}"?
   - true = text is genuinely about {topic}, discusses {topic} concepts/ideas
   - false = text is about a DIFFERENT subject (even if it uses similar style or tone)
   IMPORTANT: If the text discusses cooking, relationships, sports, politics, etc. instead of {topic}, answer FALSE even if style matches

2. style_match (0.0-1.0): How well does the text exhibit the "{style_name}" style?
   - 0.0-0.2 = opposite or completely absent style
   - 0.3-0.5 = weak hints of the style, mostly neutral
   - 0.5-0.7 = moderate style presence, noticeable but not dominant
   - 0.7-0.9 = strong style presence throughout
   - 0.9-1.0 = exceptional, perfectly captures the style

3. topic_depth (0.0-1.0): How deeply does the text engage with "{topic}"?
   - 0.0-0.2 = barely mentions topic, superficial
   - 0.3-0.5 = surface-level discussion
   - 0.5-0.7 = solid coverage of topic
   - 0.7-0.9 = detailed, insightful treatment
   - 0.9-1.0 = expert-level depth, comprehensive

4. topic_style_coherence (0.0-1.0): How naturally does the style integrate with the topic?
   - 0.0-0.2 = style feels forced, random, or contradicts topic
   - 0.3-0.5 = style is present but disconnected from topic content
   - 0.5-0.7 = style and topic coexist reasonably
   - 0.7-0.9 = style enhances the topic, feels intentional
   - 0.9-1.0 = perfect fusion, style emerges organically from topic context

IMPORTANT: 
- Be consistent in your scoring across different texts
- Scores of 0.9+ should be RARE and reserved for truly exceptional work
- Most decent texts should score in the 0.4-0.7 range
- If topic_relevant is false, other scores don't matter (final will be 0)

OUTPUT FORMAT - respond with ONLY this JSON, no other text:
{{"topic_relevant": true/false, "style_match": 0.XX, "topic_depth": 0.XX, "topic_style_coherence": 0.XX}}"""

    messages = [
        {"role": "system", "content": "You are a precise text evaluator. Output ONLY valid JSON, no explanations."},
        {"role": "user", "content": user_content}
    ]
    return messages

# --- Endpoints ---
@app.post("/rank", response_model=RankResponse)
async def rank_texts(request: RankRequest):
    if not ml_models:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    tokenizer = ml_models["tokenizer"]
    engine = ml_models["engine"]

    # Prepare Input
    messages = format_prompt(request.style_name, request.texts, request.topic)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate
    max_tokens = config["model"].get("max_new_tokens", 512)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )
    
    request_id = str(uuid.uuid4())
    results_generator = engine.generate(prompt, sampling_params, request_id)
    
    # Get final output
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
        
    response_text = final_output.outputs[0].text
    
    # Parsing Logic
    try:
        # Extract numbers inside brackets
        matches = re.findall(r"\[(\d+)\]", response_text)
        
        # Convert to 0-based indices (prompt uses 1-based)
        ranked_indices = []
        seen = set()
        
        for m in matches:
            idx = int(m) - 1
            if 0 <= idx < len(request.texts) and idx not in seen:
                ranked_indices.append(idx)
                seen.add(idx)
        
        # If some indices are missing, append them at the end (fallback)
        all_indices = set(range(len(request.texts)))
        missing_indices = list(all_indices - seen)
        ranked_indices.extend(missing_indices)
        
        # Construct result
        ranked_texts = [request.texts[i] for i in ranked_indices]
        
        return RankResponse(
            ranked_indices=ranked_indices, 
            ranked_texts=ranked_texts,
            raw_output=response_text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse model output: {response_text}. Error: {str(e)}")

@app.post("/score", response_model=ScoreResponse)
async def score_text(request: ScoreRequest):
    """
    Score a single text on multiple criteria, returning a final 0-1 score.
    """
    if not ml_models:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    tokenizer = ml_models["tokenizer"]
    engine = ml_models["engine"]

    # Prepare Input
    messages = format_score_prompt(request.style_name, request.topic, request.text)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate
    max_tokens = 200  # JSON output is short
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )
    
    request_id = str(uuid.uuid4())
    results_generator = engine.generate(prompt, sampling_params, request_id)
    
    # Get final output
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
        
    response_text = final_output.outputs[0].text.strip()
    
    # Parse JSON output
    try:
        # Try to extract JSON from response (model might add extra text)
        json_match = re.search(r'\{[^}]+\}', response_text)
        if json_match:
            json_str = json_match.group()
        else:
            json_str = response_text
            
        data = json.loads(json_str)
        
        # Extract and validate scores
        topic_relevant = bool(data.get("topic_relevant", False))
        
        def clamp_score(val, default=0.5):
            try:
                v = float(val)
                return max(0.0, min(1.0, v))
            except (ValueError, TypeError):
                return default
        
        style_match = clamp_score(data.get("style_match", 0.5))
        topic_depth = clamp_score(data.get("topic_depth", 0.5))
        topic_style_coherence = clamp_score(data.get("topic_style_coherence", 0.5))
        
        # Build criteria object
        criteria = CriteriaScores(
            topic_relevant=topic_relevant,
            style_match=style_match,
            topic_depth=topic_depth,
            topic_style_coherence=topic_style_coherence
        )
        
        # Compute final score
        scores_dict = {
            "style_match": style_match,
            "topic_depth": topic_depth,
            "topic_style_coherence": topic_style_coherence
        }
        final_score = compute_final_score(scores_dict, topic_relevant)
        
        return ScoreResponse(
            final_score=final_score,
            criteria=criteria,
            raw_output=response_text
        )
        
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to parse JSON from model output: {response_text}. Error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing score: {response_text}. Error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    host = config["server"].get("host", "0.0.0.0")
    port = config["server"].get("port", 8000)
    uvicorn.run(app, host=host, port=port)
