#!/usr/bin/env python3
"""
Unified from-folder runner with direct OpenAI-compatible API calls.

Modes:
  - scoring: writes scored.csv + scored_detailed.csv
  - ranking: writes ranked.csv
  - both: scoring then ranking
"""

import argparse
import asyncio
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
import yaml

DEFAULT_CONCURRENCY = 16


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config() -> dict:
    cfg_path = project_root() / "config.yaml"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_api_config(config: dict, args: argparse.Namespace) -> dict:
    api_cfg = config.get("api", {})
    model_cfg = config.get("model", {})

    base_url = args.base_url or os.getenv("OPENAI_BASE_URL") or api_cfg.get("base_url", "")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or api_cfg.get("api_key", "")
    model = args.model or os.getenv("OPENAI_MODEL") or api_cfg.get("model") or model_cfg.get("name", "")
    timeout = args.timeout if args.timeout is not None else int(api_cfg.get("timeout_seconds", 120))

    if not base_url:
        raise ValueError("base_url is required (use --base-url, OPENAI_BASE_URL, or config.yaml api.base_url)")
    if not model:
        raise ValueError("model is required (use --model, OPENAI_MODEL, or config.yaml api/model)")

    return {
        "base_url": base_url.rstrip("/"),
        "api_key": api_key,
        "model": model,
        "timeout": max(1, int(timeout)),
        "max_new_tokens": int(model_cfg.get("max_new_tokens", 512)),
    }


def convert_folder_to_csv(folder_path: str) -> tuple[list[str], list[list[str]]]:
    txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])

    if not txt_files:
        print(f"No .txt files found in {folder_path}")
        return [], []

    print(f"Found {len(txt_files)} files: {', '.join(txt_files)}")
    column_names = [os.path.splitext(f)[0] for f in txt_files]

    file_contents: Dict[str, List[str]] = {}
    max_lines = 0
    for txt_file in txt_files:
        filepath = os.path.join(folder_path, txt_file)
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            col_name = os.path.splitext(txt_file)[0]
            file_contents[col_name] = lines
            max_lines = max(max_lines, len(lines))

    print(f"Max lines per file: {max_lines}")

    rows: List[List[str]] = []
    for i in range(max_lines):
        row = []
        for col in column_names:
            lines = file_contents[col]
            row.append(lines[i] if i < len(lines) else "")
        rows.append(row)

    return column_names, rows


def compute_final_score(scores: Dict[str, float], topic_relevant: bool) -> float:
    if not topic_relevant:
        return 0.0

    style = scores["style_match"]
    depth = scores["topic_depth"]
    coherence = scores["topic_style_coherence"]

    arithmetic_mean = style * 0.25 + depth * 0.25 + coherence * 0.50
    geometric_mean = (style * depth * coherence) ** (1 / 3)
    final = 0.6 * arithmetic_mean + 0.4 * geometric_mean
    return round(max(0.0, min(1.0, final)), 3)


def format_rank_prompt(style_name: str, texts: List[str], topic: Optional[str] = None) -> List[dict]:
    passages_text = ""
    for idx, text in enumerate(texts):
        passages_text += f"[{idx + 1}] {text}\n"

    num = len(texts)
    if topic:
        user_content = f"""I will provide you with {num} texts, each indicated by a number identifier [].
Rank the texts based on how well they match the style: \"{style_name}\" AND the topic: \"{topic}\".

Ranking Criteria:
1. Best: The text clearly matches BOTH the style \"{style_name}\" AND the topic \"{topic}\".
2. Medium: The text matches EITHER the style \"{style_name}\" OR the topic \"{topic}\", but not both. (e.g. correct style but wrong topic, or correct topic but wrong style). These should be ranked similarly.
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
        user_content = f"""I will provide you with {num} texts, each indicated by a number identifier [].
Rank the texts based on how well they match the style: \"{style_name}\".

Texts:
{passages_text}

Ranking Task: Rank the {num} texts above based on their adherence to the style \"{style_name}\".
The texts should be listed in descending order (best match first) using their identifiers.
The output format must be strictly: [best_id] > [second_best_id] > ...
Example: [1] > [3] > [2]
Do not provide any explanation, only the ranking.
"""

    return [
        {
            "role": "system",
            "content": "You are an expert literary critic and style analyzer. Your task is to rank texts based on specific stylistic criteria.",
        },
        {"role": "user", "content": user_content},
    ]


def format_score_prompt(style_name: str, topic: str, text: str) -> List[dict]:
    user_content = f"""Evaluate this text on multiple criteria. 
Topic required: \"{topic}\"
Style required: \"{style_name}\"

TEXT TO EVALUATE:
\"\"\"{text}\"\"\"

EVALUATION CRITERIA:

1. topic_relevant (true/false): Is this text ACTUALLY ABOUT \"{topic}\"?
   - true = text is genuinely about {topic}, discusses {topic} concepts/ideas
   - false = text is about a DIFFERENT subject (even if it uses similar style or tone)
   IMPORTANT: If the text discusses cooking, relationships, sports, politics, etc. instead of {topic}, answer FALSE even if style matches

2. style_match (0.0-1.0): How well does the text exhibit the \"{style_name}\" style?
   - 0.0-0.2 = opposite or completely absent style
   - 0.3-0.5 = weak hints of the style, mostly neutral
   - 0.5-0.7 = moderate style presence, noticeable but not dominant
   - 0.7-0.9 = strong style presence throughout
   - 0.9-1.0 = exceptional, perfectly captures the style

3. topic_depth (0.0-1.0): How deeply does the text engage with \"{topic}\"?
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
{{\"topic_relevant\": true/false, \"style_match\": 0.XX, \"topic_depth\": 0.XX, \"topic_style_coherence\": 0.XX}}"""

    return [
        {
            "role": "system",
            "content": "You are a precise text evaluator. Output ONLY valid JSON, no explanations.",
        },
        {"role": "user", "content": user_content},
    ]


async def call_chat_completion(
    session: aiohttp.ClientSession,
    api_cfg: dict,
    messages: List[dict],
    max_tokens: int,
) -> str:
    payload = {
        "model": api_cfg["model"],
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if api_cfg["api_key"]:
        headers["Authorization"] = f"Bearer {api_cfg['api_key']}"

    url = f"{api_cfg['base_url']}/chat/completions"
    timeout = aiohttp.ClientTimeout(total=api_cfg["timeout"])
    async with session.post(url, headers=headers, json=payload, timeout=timeout) as response:
        response.raise_for_status()
        data = await response.json()

    return data["choices"][0]["message"]["content"]


def parse_rank_response(response_text: str, texts: List[str]) -> List[int]:
    matches = re.findall(r"\[(\d+)\]", response_text)
    ranked_indices: List[int] = []
    seen = set()

    for m in matches:
        idx = int(m) - 1
        if 0 <= idx < len(texts) and idx not in seen:
            ranked_indices.append(idx)
            seen.add(idx)

    all_indices = set(range(len(texts)))
    missing_indices = list(all_indices - seen)
    ranked_indices.extend(missing_indices)
    return ranked_indices


def clamp_score(value, default=0.5):
    try:
        v = float(value)
        return max(0.0, min(1.0, v))
    except (ValueError, TypeError):
        return default


def parse_score_response(response_text: str) -> dict:
    json_match = re.search(r"\{[^}]+\}", response_text)
    json_str = json_match.group() if json_match else response_text
    data = json.loads(json_str)

    topic_relevant = bool(data.get("topic_relevant", False))
    style_match = clamp_score(data.get("style_match", 0.5))
    topic_depth = clamp_score(data.get("topic_depth", 0.5))
    topic_style_coherence = clamp_score(data.get("topic_style_coherence", 0.5))

    final_score = compute_final_score(
        {
            "style_match": style_match,
            "topic_depth": topic_depth,
            "topic_style_coherence": topic_style_coherence,
        },
        topic_relevant,
    )

    return {
        "final_score": final_score,
        "topic_relevant": topic_relevant,
        "style_match": style_match,
        "topic_depth": topic_depth,
        "topic_style_coherence": topic_style_coherence,
        "raw_output": response_text,
    }


async def score_text_async(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_cfg: dict,
    style: str,
    topic: str,
    text: str,
    row_idx: int,
    col_idx: int,
    col_name: str,
    progress: dict,
) -> dict:
    async with semaphore:
        try:
            messages = format_score_prompt(style, topic, text)
            response_text = await call_chat_completion(session, api_cfg, messages, max_tokens=200)
            parsed = parse_score_response(response_text)

            progress["done"] += 1
            print(
                f"  [{progress['done']}/{progress['total']}] "
                f"Row {row_idx + 1}, {col_name}: {parsed['final_score']:.3f}"
            )
            return {
                "row_idx": row_idx,
                "col_idx": col_idx,
                "text": text,
                **parsed,
            }
        except Exception as e:
            progress["done"] += 1
            print(f"  [{progress['done']}/{progress['total']}] Error at row {row_idx + 1}, {col_name}: {e}")
            return {
                "row_idx": row_idx,
                "col_idx": col_idx,
                "text": text,
                "final_score": "ERROR",
                "topic_relevant": "",
                "style_match": "",
                "topic_depth": "",
                "topic_style_coherence": "",
                "raw_output": "",
            }


def print_statistics(results, column_names):
    all_scores = []
    col_scores = {col: [] for col in column_names}

    for row in results:
        for col_idx, r in enumerate(row):
            if r and isinstance(r.get("final_score"), float):
                score = r["final_score"]
                all_scores.append(score)
                col_scores[column_names[col_idx]].append(score)

    if not all_scores:
        return

    print("\n" + "=" * 50)
    print("STATISTICS")
    print("=" * 50)
    print(f"\nOverall ({len(all_scores)} texts):")
    print(f"  Mean: {sum(all_scores) / len(all_scores):.3f}")
    print(f"  Min:  {min(all_scores):.3f}")
    print(f"  Max:  {max(all_scores):.3f}")

    print("\nPer column:")
    for col in column_names:
        scores = col_scores[col]
        if scores:
            mean = sum(scores) / len(scores)
            print(
                f"  {col}: mean={mean:.3f}, min={min(scores):.3f}, "
                f"max={max(scores):.3f}, n={len(scores)}"
            )


async def run_scoring_mode(
    api_cfg: dict,
    column_names: List[str],
    rows: List[List[str]],
    topic: str,
    style: str,
    output_prefix: str,
    concurrency: int,
) -> None:
    num_cols = len(column_names)
    num_rows = len(rows)

    tasks_info = []
    for row_idx, row in enumerate(rows):
        for col_idx, text in enumerate(row):
            if text.strip():
                tasks_info.append((row_idx, col_idx, text, column_names[col_idx]))

    total_texts = len(tasks_info)
    print(f"\nScoring {total_texts} texts ({num_rows} rows x {num_cols} columns)")
    print(f"Topic: {topic}, Style: {style}")
    print(f"Concurrency: {concurrency} parallel requests")

    results = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    progress = {"done": 0, "total": total_texts}
    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        tasks = [
            score_text_async(
                session,
                semaphore,
                api_cfg,
                style,
                topic,
                text,
                row_idx,
                col_idx,
                col_name,
                progress,
            )
            for row_idx, col_idx, text, col_name in tasks_info
        ]
        completed = await asyncio.gather(*tasks)

    for item in completed:
        results[item["row_idx"]][item["col_idx"]] = item

    detailed_header = []
    for col in column_names:
        detailed_header.extend(
            [
                col,
                f"{col}_score",
                f"{col}_topic_rel",
                f"{col}_style",
                f"{col}_depth",
                f"{col}_coherence",
            ]
        )

    detailed_rows = [detailed_header]
    for row_idx in range(num_rows):
        out_row = []
        for col_idx in range(num_cols):
            r = results[row_idx][col_idx]
            if r is None:
                out_row.extend(["", "", "", "", "", ""])
                continue

            fs = r["final_score"]
            out_row.extend(
                [
                    r["text"],
                    f"{fs:.3f}" if isinstance(fs, float) else fs,
                    "1" if r["topic_relevant"] else "0" if r["topic_relevant"] is False else "",
                    f"{r['style_match']:.2f}" if isinstance(r["style_match"], float) else "",
                    f"{r['topic_depth']:.2f}" if isinstance(r["topic_depth"], float) else "",
                    f"{r['topic_style_coherence']:.2f}" if isinstance(r["topic_style_coherence"], float) else "",
                ]
            )
        detailed_rows.append(out_row)

    detailed_path = f"{output_prefix}_detailed.csv"
    with open(detailed_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(detailed_rows)
    print(f"Saved detailed results: {detailed_path}")

    simple_header = [f"{col}_score" for col in column_names]
    simple_rows = [simple_header]
    for row_idx in range(num_rows):
        out_row = []
        for col_idx in range(num_cols):
            r = results[row_idx][col_idx]
            if r is None:
                out_row.append("")
            else:
                fs = r["final_score"]
                out_row.append(f"{fs:.3f}" if isinstance(fs, float) else fs)
        simple_rows.append(out_row)

    simple_path = f"{output_prefix}.csv"
    with open(simple_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(simple_rows)
    print(f"Saved results: {simple_path}")

    print_statistics(results, column_names)


async def run_ranking_mode(
    api_cfg: dict,
    column_names: List[str],
    rows: List[List[str]],
    topic: Optional[str],
    style: str,
    output_path: str,
) -> None:
    num_cols = len(column_names)
    num_rows = len(rows)

    print(f"Topic: '{topic}', Style: '{style}'")
    print(f"Ranking {num_rows} rows x {num_cols} columns")

    output_rows = [column_names]

    async with aiohttp.ClientSession() as session:
        for row_idx, row in enumerate(rows):
            texts = [cell for cell in row if cell.strip()]
            if len(texts) == 0:
                print(f"Skipping empty row {row_idx + 1}")
                continue

            try:
                messages = format_rank_prompt(style, texts, topic)
                response_text = await call_chat_completion(
                    session,
                    api_cfg,
                    messages,
                    max_tokens=api_cfg["max_new_tokens"],
                )
                ranked_indices = parse_rank_response(response_text, texts)

                ranks = [0] * len(texts)
                for rank, original_idx in enumerate(ranked_indices):
                    ranks[original_idx] = rank + 1

                output_rows.append(ranks)
                print(f"Row {row_idx + 1} processed.")
            except Exception as e:
                print(f"Error processing row {row_idx + 1}: {e}")
                output_rows.append(["ERROR"] * len(texts))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(output_rows)
    print(f"Results saved to: {output_path}")


def build_default_paths(folder_path: str) -> tuple[str, str]:
    folder_name = os.path.basename(os.path.normpath(folder_path))
    output_dir = os.path.join("results", folder_name)
    return os.path.join(output_dir, "scored"), os.path.join(output_dir, "ranked.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified from-folder runner (direct OpenAI-compatible API, no local ranking server)",
    )
    parser.add_argument("folder_path", help="Path to folder with .txt files")
    parser.add_argument(
        "--mode",
        choices=["scoring", "ranking", "both"],
        default="both",
        help="Run mode (default: both)",
    )
    parser.add_argument("--topic", "-t", default=None, help="Topic name")
    parser.add_argument("--style", "-s", required=True, help="Style name")
    parser.add_argument("--concurrency", "-c", type=int, default=DEFAULT_CONCURRENCY)

    parser.add_argument("--scoring-output", default=None, help="Scoring output prefix (e.g. results/x/scored)")
    parser.add_argument("--ranking-output", default=None, help="Ranking output file (e.g. results/x/ranked.csv)")

    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL, usually .../v1")
    parser.add_argument("--api-key", default=None, help="API key (optional for local endpoints)")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--timeout", type=int, default=None, help="Request timeout in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.folder_path):
        print(f"Error: {args.folder_path} is not a directory")
        return

    if args.mode in ("scoring", "both") and not args.topic:
        print("Error: --topic is required for scoring mode")
        return

    column_names, rows = convert_folder_to_csv(args.folder_path)
    if not column_names:
        return

    default_scored_prefix, default_ranked_path = build_default_paths(args.folder_path)
    scoring_output = args.scoring_output or default_scored_prefix
    ranking_output = args.ranking_output or default_ranked_path

    os.makedirs(os.path.dirname(scoring_output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(ranking_output) or ".", exist_ok=True)

    try:
        cfg = load_config()
        api_cfg = resolve_api_config(cfg, args)
    except Exception as e:
        print(f"Config error: {e}")
        return

    print("=" * 60)
    print("OPENAI-COMPATIBLE RUNNER")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Model: {api_cfg['model']}")
    print(f"Base URL: {api_cfg['base_url']}")
    print(f"Folder: {args.folder_path}")

    if args.mode in ("scoring", "both"):
        asyncio.run(
            run_scoring_mode(
                api_cfg,
                column_names,
                rows,
                args.topic,
                args.style,
                scoring_output,
                args.concurrency,
            )
        )

    if args.mode in ("ranking", "both"):
        asyncio.run(
            run_ranking_mode(
                api_cfg,
                column_names,
                rows,
                args.topic,
                args.style,
                ranking_output,
            )
        )


if __name__ == "__main__":
    main()
