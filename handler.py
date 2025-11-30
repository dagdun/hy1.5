import base64
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import runpod

REPO_DIR = Path(os.getenv("HY_REPO_DIR", "/opt/hunyuan/HunyuanVideo-1.5"))
GENERATE_SCRIPT = REPO_DIR / "generate.py"
DEFAULT_MODEL_PATH = os.getenv("HY_MODEL_PATH", str(REPO_DIR / "ckpts"))

BOOL_FIELDS = {
    "sr",
    "save_pre_sr_video",
    "rewrite",
    "cfg_distilled",
    "sparse_attn",
    "offloading",
    "group_offloading",
    "overlap_group_offloading",
    "use_sageattn",
    "enable_torch_compile",
    "enable_cache",
    "save_generation_config",
}

INT_FIELDS = {
    "num_inference_steps",
    "video_length",
    "seed",
    "cache_start_step",
    "cache_end_step",
    "total_steps",
    "cache_step_interval",
}

DEFAULT_ARGS: Dict[str, Any] = {
    "prompt": None,
    "negative_prompt": "",
    "resolution": os.getenv("HY_DEFAULT_RESOLUTION", "480p"),
    "model_path": DEFAULT_MODEL_PATH,
    "aspect_ratio": "16:9",
    "num_inference_steps": 50,
    "video_length": 121,
    "sr": True,
    "save_pre_sr_video": False,
    "rewrite": False,
    "cfg_distilled": False,
    "sparse_attn": False,
    "offloading": True,
    "group_offloading": None,
    "overlap_group_offloading": True,
    "dtype": "bf16",
    "seed": 123,
    "image_path": None,
    "output_path": None,
    "use_sageattn": False,
    "sage_blocks_range": "0-53",
    "enable_torch_compile": False,
    "enable_cache": False,
    "cache_type": "deepcache",
    "no_cache_block_id": "53",
    "cache_start_step": 11,
    "cache_end_step": 45,
    "total_steps": 50,
    "cache_step_interval": 4,
    "save_generation_config": True,
}

VALID_RESOLUTIONS = {"480p", "720p"}


def _parse_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"Invalid boolean for '{field}': {value!r}")


def _parse_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for '{field}': {value!r}") from exc


def _sanitize_path(value: Any) -> Any:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() == "none":
        return None
    return normalized


def _decode_image_base64(b64_data: str, destination: Path) -> Path:
    payload = b64_data.strip()
    if "," in payload and payload.split(",", 1)[0].startswith("data:"):
        payload = payload.split(",", 1)[1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as file:
        file.write(base64.b64decode(payload))
    return destination


def _prepare_arguments(job_input: Dict[str, Any]) -> Tuple[List[str], Path, Path]:
    args = DEFAULT_ARGS.copy()
    for key, value in job_input.items():
        if key in args and value is not None:
            args[key] = value

    prompt = job_input.get("prompt") or args.get("prompt")
    if not prompt:
        raise ValueError("`prompt` is required.")
    args["prompt"] = prompt

    if args["resolution"] not in VALID_RESOLUTIONS:
        raise ValueError(f"`resolution` must be one of {sorted(VALID_RESOLUTIONS)}.")

    for field in BOOL_FIELDS:
        if args.get(field) is None:
            continue
        args[field] = _parse_bool(args[field], field)

    for field in INT_FIELDS:
        if args.get(field) is None:
            continue
        args[field] = _parse_int(args[field], field)

    args["image_path"] = _sanitize_path(args.get("image_path"))

    model_path = Path(args["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"model_path does not exist: {model_path}")

    if not GENERATE_SCRIPT.exists():
        raise FileNotFoundError(f"Could not locate generate.py at {GENERATE_SCRIPT}")

    temp_dir = Path(tempfile.mkdtemp(prefix="hunyuan_job_"))
    output_path = temp_dir / "output.mp4"
    args["output_path"] = str(output_path)

    image_temp_path = None
    image_b64 = job_input.get("image_base64")
    if image_b64:
        image_temp_path = temp_dir / "reference.png"
        _decode_image_base64(image_b64, image_temp_path)
        args["image_path"] = str(image_temp_path)

    cli = ["python", str(GENERATE_SCRIPT), "--prompt", args["prompt"], "--resolution", args["resolution"], "--model_path", args["model_path"]]

    for key, value in args.items():
        if key in {"prompt", "resolution", "model_path"}:
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        cli.extend([f"--{key}", str(value)])

    return cli, output_path, temp_dir


def _run_generate(cmd: List[str]) -> Tuple[str, str]:
    completed = subprocess.run(
        cmd,
        cwd=REPO_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"generate.py exited with status {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed.stdout, completed.stderr


def _encode_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Expected output video missing: {path}")
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def run_infer(job: Dict[str, Any]) -> Dict[str, Any]:
    job_input = job.get("input") or {}
    cmd: List[str] = []
    output_path: Path
    temp_dir: Path = Path()
    try:
        cmd, output_path, temp_dir = _prepare_arguments(job_input)
        stdout, stderr = _run_generate(cmd)
        video_base64 = _encode_file(output_path)
        response = {
            "video_base64": video_base64,
            "output_path": str(output_path),
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
        }
        config_path = output_path.with_name(output_path.stem + "_config.json")
        if config_path.exists():
            response["config_json"] = config_path.read_text(encoding="utf-8")
        return response
    except Exception as exc:
        return {
            "error": str(exc),
            "extra": {
                "command": cmd,
                "repo_dir_exists": REPO_DIR.exists(),
            },
        }
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


runpod.serverless.start({"handler": run_infer})
