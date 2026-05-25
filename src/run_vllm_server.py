"""
Запускает vLLM OpenAI-compatible сервер на основе config.yaml.

После запуска endpoint доступен по адресу:
    http://{server.host}:{server.port}/v1

Этот URL можно использовать как api.base_url для process_pipeline.py
и run_openai_from_folder.py.
"""

import argparse
import os
import sys
from pathlib import Path

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config() -> dict:
    cfg_path = project_root() / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f) or {}


def parse_device_config(device_cfg):
    """Вернуть (CUDA_VISIBLE_DEVICES, tensor_parallel_size). См. rank_server.py."""
    if device_cfg is None or device_cfg == "cuda":
        return None, 1

    gpu_ids = []
    if isinstance(device_cfg, list):
        for item in device_cfg:
            if isinstance(item, int):
                gpu_ids.append(str(item))
            elif isinstance(item, str):
                if "cuda:" in item:
                    gpu_ids.append(item.split(":")[1])
                else:
                    gpu_ids.append(item.strip())
    elif isinstance(device_cfg, str):
        if "cuda:" in device_cfg:
            ids_part = device_cfg.split("cuda:")[-1]
            gpu_ids = [x.strip() for x in ids_part.split(",")]
        elif "," in device_cfg:
            gpu_ids = [x.strip() for x in device_cfg.split(",")]
        else:
            gpu_ids = [device_cfg.strip()]
    elif isinstance(device_cfg, int):
        gpu_ids = [str(device_cfg)]

    if not gpu_ids:
        return None, 1

    return ",".join(gpu_ids), len(gpu_ids)


def main():
    parser = argparse.ArgumentParser(description="Запуск vLLM OpenAI-compatible сервера из config.yaml")
    parser.add_argument("--host", default=None, help="Переопределить server.host")
    parser.add_argument("--port", type=int, default=None, help="Переопределить server.port")
    parser.add_argument("--model", default=None, help="Переопределить model.name")
    parser.add_argument("--served-model-name", default=None,
                        help="Имя модели, под которым она будет видна в API (по умолчанию = model.name)")
    parser.add_argument("--dtype", default="auto", help="dtype для vLLM (auto, bfloat16, float16, ...)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None,
                        help="Доля памяти GPU для vLLM (0..1). По умолчанию vLLM сам решает.")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="Доп. аргументы, пробрасываются как есть в vllm serve")
    args = parser.parse_args()

    cfg = load_config()
    model_cfg = cfg.get("model", {})
    api_cfg = cfg.get("api", {})
    server_cfg = cfg.get("server", {})

    model_name = args.model or model_cfg.get("name")
    if not model_name:
        raise SystemExit("model.name не задан в config.yaml и --model не передан")

    served_name = args.served_model_name or api_cfg.get("model") or model_name

    host = args.host or server_cfg.get("host", "0.0.0.0")
    port = int(args.port if args.port is not None else server_cfg.get("port", 8000))

    api_key = args.api_key if args.api_key is not None else (api_cfg.get("api_key") or "")
    max_model_len = model_cfg.get("max_context_length")
    device_cfg = model_cfg.get("device", "cuda")
    cuda_visible, tp_size = parse_device_config(device_cfg)

    env = os.environ.copy()
    if cuda_visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
        print(f"[INFO] CUDA_VISIBLE_DEVICES={cuda_visible}")
    print(f"[INFO] tensor_parallel_size={tp_size}")
    print(f"[INFO] model={model_name}  served-as={served_name}")
    print(f"[INFO] listening on http://{host}:{port}/v1")
    if api_key:
        print(f"[INFO] API key required (len={len(api_key)})")
    else:
        print("[INFO] API key not set — endpoint открыт без авторизации")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--served-model-name", served_name,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tp_size),
        "--dtype", args.dtype,
        "--trust-remote-code",
    ]
    if max_model_len:
        cmd += ["--max-model-len", str(int(max_model_len))]
    if api_key:
        cmd += ["--api-key", str(api_key)]
    if args.gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    if args.extra:
        cmd += args.extra

    print("[INFO] exec:", " ".join(cmd))
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
