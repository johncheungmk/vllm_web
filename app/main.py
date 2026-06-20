from __future__ import annotations

import json
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

import requests
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("VLLM_UI_DATA", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROFILES_PATH = DATA_DIR / "profiles.json"
EXAMPLE_PROFILES_PATH = ROOT / "data" / "profiles.example.json"
LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
PUBLIC_BIND_HOSTS = {"0.0.0.0", "::", ""}
VALID_DEPLOYMENT_MODES = {
    "local", "single_node_multi_gpu", "ray_cluster", "data_parallel_internal",
    "data_parallel_external", "data_parallel_hybrid", "expert_parallel_moe", "spark_ray",
    "ray_symmetric_run",
}
VALID_DISTRIBUTED_BACKENDS = {None, "", "mp", "ray", "uni", "external_launcher"}
VALID_UVICORN_LEVELS = {"critical", "error", "warning", "info", "debug", "trace"}
VALID_DTYPES = {"auto", "half", "float16", "bfloat16", "float", "float32"}
VALID_QUANTIZATIONS = {
    "", "awq", "gptq", "squeezellm", "bitsandbytes", "fp8", "compressed-tensors",
    "gguf", "marlin", "awq_marlin", "gptq_marlin", "experts_int8", "modelopt",
    "nvfp4", "tpu_int8", "inc", "fbgemm_fp8",
}
VALID_TASKS = {"auto", "generate", "embedding", "embed", "classify", "score", "reward", "transcription"}
VALID_RUNNERS = {"auto", "generate", "pooling", "draft"}
VALID_CONVERTS = {"auto", "none", "embed", "classify"}
VALID_LOAD_FORMATS = {
    "auto", "pt", "safetensors", "npcache", "dummy", "tensorizer", "bitsandbytes",
    "sharded_state", "gguf", "mistral", "runai_streamer", "fastsafetensors",
}
VALID_GENERATION_CONFIGS = {"", "auto", "vllm", "custom"}
VALID_GUIDED_DECODING_BACKENDS = {"", "auto", "xgrammar", "outlines", "guidance"}
VALID_TOKENIZER_MODES = {"auto", "slow", "mistral", "custom"}
VALID_KV_CACHE_DTYPES = {
    "", "auto", "bfloat16", "float16", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_ds_mla",
    "fp8_inc", "fp8_per_token_head", "int8_per_token_head", "nvfp4", "turboquant_3bit_nc",
    "turboquant_4bit_nc", "turboquant_k3v4_nc", "turboquant_k8v4",
}
VALID_HASH_ALGOS = {"", "sha256", "sha256_cbor", "xxhash", "xxhash_cbor"}
VALID_PERFORMANCE_MODES = {"", "balanced", "interactivity", "throughput"}
VALID_MAMBA_CACHE_DTYPES = {"", "auto", "bfloat16", "float16", "float32"}
VALID_MAMBA_CACHE_MODES = {"", "none", "all", "align"}
VALID_KV_OFFLOADING_BACKENDS = {"", "native", "lmcache"}
VALID_OFFLOAD_BACKENDS = {"", "auto", "prefetch", "uva"}
VALID_SPEC_METHODS = {
    "", "draft_model", "ngram", "ngram_gpu", "eagle", "eagle3", "medusa", "suffix", "mtp",
    "mlp_speculator", "custom_class", "deepseek_mtp", "dflash", "ernie_mtp", "exaone4_5_mtp",
    "exaone_moe_mtp", "extract_hidden_states", "gemma4_mtp", "glm4_moe_lite_mtp",
    "glm4_moe_mtp", "glm_ocr_mtp", "hy_v3_mtp", "longcat_flash_mtp", "mimo_mtp",
    "mimo_v2_mtp", "nemotron_h_mtp", "pangu_ultra_moe_mtp", "qwen3_5_mtp",
    "qwen3_next_mtp", "step3p5_mtp",
}

FALLBACK_CHOICES: Dict[str, List[str]] = {
    "dtype": sorted(VALID_DTYPES),
    "quantization": ["", "awq", "gptq", "squeezellm", "bitsandbytes", "fp8", "compressed-tensors", "gguf", "marlin", "awq_marlin", "gptq_marlin", "experts_int8", "modelopt", "nvfp4", "tpu_int8", "inc", "fbgemm_fp8"],
    "runner": ["auto", "generate", "pooling", "draft"],
    "convert": ["auto", "none", "embed", "classify"],
    "task": sorted(VALID_TASKS),
    "load_format": ["auto", "pt", "safetensors", "npcache", "dummy", "tensorizer", "bitsandbytes", "sharded_state", "gguf", "mistral", "runai_streamer", "fastsafetensors"],
    "uvicorn_log_level": ["critical", "error", "warning", "info", "debug", "trace"],
    "generation_config": ["auto", "vllm", "custom"],
    "guided_decoding_backend": ["", "auto", "xgrammar", "outlines", "guidance"],
    "tokenizer_mode": ["auto", "slow", "mistral", "custom"],
    "kv_cache_dtype": ["auto", "bfloat16", "float16", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_ds_mla", "fp8_inc", "fp8_per_token_head", "int8_per_token_head", "nvfp4", "turboquant_3bit_nc", "turboquant_4bit_nc", "turboquant_k3v4_nc", "turboquant_k8v4"],
    "prefix_caching_hash_algo": ["sha256", "sha256_cbor", "xxhash", "xxhash_cbor"],
    "performance_mode": ["balanced", "interactivity", "throughput"],
    "mamba_cache_dtype": ["auto", "bfloat16", "float16", "float32"],
    "mamba_cache_mode": ["none", "all", "align"],
    "kv_offloading_backend": ["native", "lmcache"],
    "offload_backend": ["auto", "prefetch", "uva"],
    "distributed_executor_backend": ["", "mp", "ray", "external_launcher", "uni"],
    "speculative_method": ["", "draft_model", "eagle", "eagle3", "mtp", "ngram", "ngram_gpu", "suffix", "medusa", "mlp_speculator", "dflash", "custom_class"],
}

OPTION_TO_FLAG = {
    "runner": "--runner",
    "convert": "--convert",
    "task": "--task",
    "dtype": "--dtype",
    "quantization": "--quantization",
    "load_format": "--load-format",
    "tokenizer_mode": "--tokenizer-mode",
    "generation_config": "--generation-config",
    "guided_decoding_backend": "--guided-decoding-backend",
    "kv_cache_dtype": "--kv-cache-dtype",
    "prefix_caching_hash_algo": "--prefix-caching-hash-algo",
    "performance_mode": "--performance-mode",
    "mamba_cache_dtype": "--mamba-cache-dtype",
    "mamba_cache_mode": "--mamba-cache-mode",
    "kv_offloading_backend": "--kv-offloading-backend",
    "offload_backend": "--offload-backend",
    "distributed_executor_backend": "--distributed-executor-backend",
}


class VllmOptionsRegistry:
    def __init__(self) -> None:
        self.detected = False
        self.version = ""
        self.help_text = ""
        self.supported_flags: set[str] = set()
        self.choices: Dict[str, List[str]] = {k: list(v) for k, v in FALLBACK_CHOICES.items()}
        self.warnings: List[str] = []
        self.refresh()

    def refresh(self) -> None:
        self.warnings.clear()
        try:
            version_out = subprocess.check_output(["vllm", "--version"], text=True, timeout=5, stderr=subprocess.STDOUT)
            self.version = version_out.strip()
        except Exception:
            self.version = ""
        try:
            self.help_text = subprocess.check_output(["vllm", "serve", "--help"], text=True, timeout=10, stderr=subprocess.STDOUT)
            self.detected = True
            self.supported_flags = set(re.findall(r"(?<!\w)(--[a-zA-Z0-9][a-zA-Z0-9-]*)", self.help_text))
            self._parse_choices()
        except Exception as exc:
            self.detected = False
            self.help_text = ""
            self.supported_flags = set()
            self.warnings.append(f"vLLM not detected; using fallback option list. {exc}")

    def _parse_choices(self) -> None:
        for field_name, flag in OPTION_TO_FLAG.items():
            pattern = rf"{re.escape(flag)}[^\n]*\{{([^}}]+)\}}"
            match = re.search(pattern, self.help_text)
            if not match:
                continue
            parsed = [item.strip() for item in match.group(1).split(",") if item.strip()]
            if parsed:
                existing = self.choices.get(field_name, [])
                self.choices[field_name] = list(dict.fromkeys(existing + parsed))
        # Speculative method choices are often only documented in prose or JSON examples.
        spec_pattern = r"\b([a-zA-Z0-9_]+_mtp|ngram_gpu|mlp_speculator|draft_model|eagle3?|medusa|suffix|mtp|dflash|extract_hidden_states|custom_class)\b"
        for method in re.findall(spec_pattern, self.help_text):
            if method not in self.choices["speculative_method"]:
                self.choices["speculative_method"].append(method)

    def flag_supported(self, flag: str) -> bool:
        return (not self.detected) or flag in self.supported_flags

    def unsupported_ui_fields(self) -> List[str]:
        unsupported = []
        for field_name, flag in OPTION_TO_FLAG.items():
            if self.detected and flag not in self.supported_flags:
                unsupported.append(field_name)
        return unsupported

    def payload(self) -> Dict[str, Any]:
        warnings = list(self.warnings)
        for field_name in self.unsupported_ui_fields():
            warnings.append(f"{OPTION_TO_FLAG[field_name]} is not supported by the installed vLLM help output.")
        return {
            "detected": self.detected,
            "version": self.version,
            "supported_cli_options": sorted(self.supported_flags),
            "choices": self.choices,
            "fallback_choices": FALLBACK_CHOICES,
            "unsupported_ui_fields": self.unsupported_ui_fields(),
            "warnings": warnings,
        }


vllm_options = VllmOptionsRegistry()


class VllmProfile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Default vLLM profile"
    ui_mode: str = "wizard"

    # Basic server/model
    model: str = "Qwen/Qwen3-8B"
    served_model_name: str = ""
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    api_key: str = ""
    uvicorn_log_level: str = "info"
    root_path: str = ""
    generation_config: str = ""  # blank, auto, vllm
    generation_config_path: str = ""

    # Model/runtime options
    dtype: str = "auto"
    quantization: str = ""
    runner: str = "auto"
    convert: str = "auto"
    task: str = ""  # compatibility field for older vLLM versions
    load_format: str = "auto"
    tokenizer_mode: str = "auto"
    guided_decoding_backend: str = ""
    kv_cache_dtype: str = ""
    max_model_len: Optional[int] = Field(default=None, ge=1)
    gpu_memory_utilization: Optional[float] = Field(default=0.90, gt=0, le=1)
    max_num_seqs: Optional[int] = Field(default=None, ge=1)
    max_num_batched_tokens: Optional[int] = Field(default=None, ge=1)
    swap_space: Optional[float] = Field(default=None, ge=0)
    cpu_offload_gb: Optional[float] = Field(default=None, ge=0)
    download_dir: str = ""
    trust_remote_code: bool = False
    enforce_eager: bool = False
    enable_prefix_caching: bool = False
    prefix_caching_hash_algo: str = ""
    performance_mode: str = ""
    mamba_cache_dtype: str = ""
    mamba_cache_mode: str = ""
    kv_offloading_backend: str = ""
    offload_backend: str = ""
    enable_chunked_prefill: bool = False
    disable_log_stats: bool = False

    # Parallel/distributed options
    tensor_parallel_size: int = Field(default=1, ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)
    distributed_executor_backend: str = ""  # blank, mp, ray, uni, external_launcher depending on vLLM version
    data_parallel_size: Optional[int] = Field(default=None, ge=1)
    data_parallel_size_local: Optional[int] = Field(default=None, ge=0)
    data_parallel_start_rank: Optional[int] = Field(default=None, ge=0)
    data_parallel_address: str = ""
    data_parallel_rpc_port: Optional[int] = Field(default=None, ge=1, le=65535)
    data_parallel_backend: str = ""
    data_parallel_hybrid_lb: bool = False
    data_parallel_external_lb: bool = False
    data_parallel_multi_port_external_lb: bool = False
    data_parallel_rank: Optional[int] = Field(default=None, ge=0)
    headless: bool = False
    api_server_count: Optional[int] = Field(default=None, ge=1)
    is_moe_model: bool = False
    external_load_balancer: bool = False
    enable_expert_parallel: bool = False
    enable_ep_weight_filter: bool = False

    # Deployment export helpers. These fields generate scripts/notes; vLLM runtime
    # arguments still come from the fields above.
    deployment_mode: str = "local"  # local, ray_cluster, ray_symmetric_run, spark_ray
    ray_head_address: str = ""
    ray_port: int = Field(default=6379, ge=1, le=65535)
    ray_num_nodes: Optional[int] = Field(default=None, ge=1)
    ray_gpus_per_node: Optional[int] = Field(default=None, ge=0)
    ray_min_nodes: Optional[int] = Field(default=None, ge=1)
    ray_node_ips: str = ""  # newline or comma separated, first entry is head
    spark_master_url: str = ""
    spark_app_name: str = "vllm-ray-cluster"
    spark_executor_instances: Optional[int] = Field(default=None, ge=1)
    spark_executor_cores: Optional[int] = Field(default=None, ge=1)
    spark_executor_gpus: Optional[int] = Field(default=None, ge=0)
    spark_submit_args: str = ""
    spark_conf_json: str = "{}"

    # Speculative decoding. Latest vLLM uses --speculative-config JSON.
    speculative_enabled: bool = False
    speculative_method: str = ""
    speculative_model: str = ""
    num_speculative_tokens: Optional[int] = Field(default=5, ge=1)
    draft_tensor_parallel_size: Optional[int] = Field(default=None, ge=1)
    prompt_lookup_min: Optional[int] = Field(default=None, ge=1)
    prompt_lookup_max: Optional[int] = Field(default=None, ge=1)
    speculative_extra_json: str = "{}"

    # Advanced
    cuda_visible_devices: str = ""
    env_json: str = "{}"
    advanced_args: str = ""
    structured_outputs_config: str = "{}"
    compilation_config: str = "{}"
    kv_transfer_config: str = "{}"
    additional_config: str = "{}"
    override_generation_config: str = "{}"
    notes: str = ""

    @field_validator(
        "id", "name", "model", "served_model_name", "host", "api_key", "uvicorn_log_level",
        "ui_mode", "root_path", "generation_config", "generation_config_path", "dtype", "quantization",
        "runner", "convert", "task", "load_format", "tokenizer_mode", "guided_decoding_backend",
        "kv_cache_dtype", "prefix_caching_hash_algo", "performance_mode", "mamba_cache_dtype",
        "mamba_cache_mode", "kv_offloading_backend", "offload_backend", "download_dir", "distributed_executor_backend",
        "data_parallel_address", "data_parallel_backend", "deployment_mode", "ray_head_address",
        "ray_node_ips", "spark_master_url", "spark_app_name", "spark_submit_args", "spark_conf_json",
        "speculative_method", "speculative_model", "speculative_extra_json", "cuda_visible_devices",
        "env_json", "advanced_args", "structured_outputs_config", "compilation_config",
        "kv_transfer_config", "additional_config", "override_generation_config", "notes",
        mode="before",
    )
    @classmethod
    def none_to_empty_string(cls, value: Any) -> Any:
        return "" if value is None else value


class ProfilePatch(BaseModel):
    profile: VllmProfile


class ProfileValidationRequest(BaseModel):
    profile: VllmProfile


class ChatRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    max_tokens: int = Field(default=128, ge=1, le=32768)
    temperature: float = Field(default=0.2, ge=0, le=2)


class ProcessState:
    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen[str]] = None
        self.profile_id: Optional[str] = None
        self.command: List[str] = []
        self.started_at: Optional[float] = None
        self.log_lines: deque[str] = deque(maxlen=2000)
        self.log_file: Optional[Path] = None
        self._lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def append_log(self, line: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        text = f"[{ts}] {line.rstrip()}"
        self.log_lines.append(text)
        if self.log_file:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(text + "\n")

    def start_reader(self) -> None:
        def _reader() -> None:
            if not self.proc or not self.proc.stdout:
                return
            try:
                for line in self.proc.stdout:
                    self.append_log(line)
            except Exception as exc:  # pragma: no cover
                self.append_log(f"log reader error: {exc}")

        self._reader_thread = threading.Thread(target=_reader, daemon=True)
        self._reader_thread.start()


state = ProcessState()
app = FastAPI(title="vLLM Web", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1", "http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(ROOT / "app" / "static")), name="static")


def load_profiles() -> List[VllmProfile]:
    if not PROFILES_PATH.exists():
        if EXAMPLE_PROFILES_PATH.exists():
            raw = json.loads(EXAMPLE_PROFILES_PATH.read_text(encoding="utf-8"))
            profiles = [VllmProfile(**item) for item in raw]
        else:
            profiles = [VllmProfile()]
        save_profiles(profiles)
        return profiles
    raw = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    return [VllmProfile(**item) for item in raw]


def save_profiles(profiles: List[VllmProfile]) -> None:
    PROFILES_PATH.write_text(
        json.dumps([p.model_dump() for p in profiles], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_profile(profile_id: str) -> VllmProfile:
    for p in load_profiles():
        if p.id == profile_id:
            return p
    raise HTTPException(status_code=404, detail="Profile not found")


def profile_export_payload(profiles: Sequence[VllmProfile]) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "app": "vllm-web",
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "profiles": [p.model_dump() for p in profiles],
    }


def add_arg(cmd: List[str], flag: str, value: Any = None, *, boolean: bool = False) -> None:
    if boolean:
        if bool(value):
            cmd.append(flag)
        return
    if value is None or value == "":
        return
    cmd.extend([flag, str(value)])


def add_supported_arg(cmd: List[str], flag: str, value: Any = None, *, boolean: bool = False, omit_values: Sequence[Any] = ()) -> None:
    if not vllm_options.flag_supported(flag):
        return
    if value in omit_values:
        return
    add_arg(cmd, flag, value, boolean=boolean)


def add_json_arg(cmd: List[str], flag: str, text: str, field_name: str) -> None:
    if not vllm_options.flag_supported(flag):
        return
    obj = parse_json_object(text, field_name)
    if obj:
        cmd.extend([flag, json.dumps(obj, separators=(",", ":"))])


def parse_json_object(text: str, field_name: str) -> Dict[str, Any]:
    text = (text or "{}").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in {field_name}: {exc}")
    if not isinstance(obj, dict):
        raise HTTPException(status_code=400, detail=f"{field_name} must be a JSON object")
    return obj


def validation_report(p: VllmProfile) -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if not p.name.strip():
        errors.append("Profile name is required.")
    if not p.model.strip():
        errors.append("Model path or Hugging Face ID is required.")
    if p.dtype not in VALID_DTYPES:
        errors.append(f"dtype must be one of: {', '.join(sorted(VALID_DTYPES))}.")
    if p.quantization not in VALID_QUANTIZATIONS:
        warnings.append("Quantization is not in the built-in list; it will be passed through as a custom value.")
    if p.runner not in VALID_RUNNERS:
        warnings.append("Runner is not in the built-in list; confirm your vLLM version supports it.")
    if p.convert not in VALID_CONVERTS:
        warnings.append("Convert is not in the built-in list; confirm your vLLM version supports it.")
    if p.task and p.task not in VALID_TASKS:
        warnings.append("Compatibility task is not in the built-in list; confirm your vLLM version supports it.")
    if p.load_format not in VALID_LOAD_FORMATS:
        warnings.append("Load format is not in the built-in list; confirm your vLLM version supports it.")
    if p.tokenizer_mode not in VALID_TOKENIZER_MODES:
        warnings.append("Tokenizer mode is not in the built-in list; confirm your vLLM version supports it.")
    if p.generation_config not in VALID_GENERATION_CONFIGS:
        warnings.append("Generation config is not in the built-in list; confirm your vLLM version supports it.")
    if p.generation_config == "custom" and not p.generation_config_path.strip():
        errors.append("Generation config custom path is required when Generation config is set to custom path.")
    if p.guided_decoding_backend not in VALID_GUIDED_DECODING_BACKENDS:
        warnings.append("Guided decoding backend is not in the built-in list; confirm your vLLM version supports it.")
    if p.kv_cache_dtype and p.kv_cache_dtype not in VALID_KV_CACHE_DTYPES:
        warnings.append("KV cache dtype is custom; confirm your vLLM version supports it.")
    if p.prefix_caching_hash_algo and p.prefix_caching_hash_algo not in VALID_HASH_ALGOS:
        warnings.append("Prefix caching hash algorithm is custom; confirm your vLLM version supports it.")
    if p.performance_mode and p.performance_mode not in VALID_PERFORMANCE_MODES:
        warnings.append("Performance mode is custom; confirm your vLLM version supports it.")
    if p.mamba_cache_dtype and p.mamba_cache_dtype not in VALID_MAMBA_CACHE_DTYPES:
        warnings.append("Mamba cache dtype is custom; confirm your vLLM version supports it.")
    if p.mamba_cache_mode and p.mamba_cache_mode not in VALID_MAMBA_CACHE_MODES:
        warnings.append("Mamba cache mode is custom; confirm your vLLM version supports it.")
    if p.kv_offloading_backend and p.kv_offloading_backend not in VALID_KV_OFFLOADING_BACKENDS:
        warnings.append("KV offloading backend is custom; confirm your vLLM version supports it.")
    if p.offload_backend and p.offload_backend not in VALID_OFFLOAD_BACKENDS:
        warnings.append("Offload backend is custom; confirm your vLLM version supports it.")
    if p.uvicorn_log_level not in VALID_UVICORN_LEVELS:
        errors.append(f"Uvicorn log level must be one of: {', '.join(sorted(VALID_UVICORN_LEVELS))}.")
    if p.distributed_executor_backend not in VALID_DISTRIBUTED_BACKENDS:
        errors.append("Distributed executor backend must be blank, mp, ray, uni, or external_launcher.")
    if p.deployment_mode not in VALID_DEPLOYMENT_MODES:
        errors.append("Deployment mode is not recognized.")
    if p.speculative_method not in VALID_SPEC_METHODS:
        warnings.append("Speculative decoding method is not in the built-in list; confirm your vLLM version supports it.")
    if p.speculative_enabled and not p.speculative_method:
        errors.append("Choose a speculative decoding method or disable speculative decoding.")

    for field_name in (
        "env_json", "speculative_extra_json", "spark_conf_json", "structured_outputs_config",
        "compilation_config", "kv_transfer_config", "additional_config", "override_generation_config",
    ):
        try:
            obj = parse_json_object(getattr(p, field_name), field_name)
        except HTTPException as exc:
            errors.append(str(exc.detail))
            continue
        if field_name == "env_json":
            for key in obj:
                if not key or not isinstance(key, str):
                    errors.append("Environment JSON keys must be non-empty strings.")
                elif "\x00" in key or "=" in key:
                    errors.append(f"Invalid environment variable name: {key!r}.")

    try:
        shlex.split(p.advanced_args or "")
    except ValueError as exc:
        errors.append(f"Advanced vLLM args cannot be parsed: {exc}.")
    try:
        shlex.split(p.spark_submit_args or "")
    except ValueError as exc:
        errors.append(f"Spark submit args cannot be parsed: {exc}.")

    if p.host in PUBLIC_BIND_HOSTS:
        warnings.append("The vLLM API host is bound publicly. Put TLS, auth, and rate limits in front of it.")
    if p.api_key and p.host in PUBLIC_BIND_HOSTS:
        warnings.append("The profile stores an API key while the vLLM API is public-bound; protect data/profiles.json.")
    if p.trust_remote_code:
        warnings.append("trust_remote_code executes model repository code. Use only with trusted model sources.")
    for field_name in vllm_options.unsupported_ui_fields():
        if getattr(p, field_name, "") not in ("", None, "auto"):
            warnings.append(f"{OPTION_TO_FLAG[field_name]} is not supported by the installed vLLM version and will be omitted.")
    if (p.tensor_parallel_size > 1 or p.pipeline_parallel_size > 1) and p.distributed_executor_backend in {None, ""}:
        warnings.append("Distributed backend is blank; vLLM will choose a backend. Set mp for single-node or ray for multi-node when you need deterministic behavior.")
    if p.pipeline_parallel_size > 1 and p.distributed_executor_backend != "ray" and p.deployment_mode in {"ray_cluster", "ray_symmetric_run", "spark_ray"}:
        warnings.append("Pipeline parallel multi-node deployments should normally use distributed executor backend ray.")
    if p.ray_num_nodes and p.ray_num_nodes > 1:
        if p.distributed_executor_backend != "ray":
            warnings.append("Ray node count is greater than 1 but distributed executor backend is not ray.")
        if p.pipeline_parallel_size != p.ray_num_nodes:
            warnings.append("For multi-node vLLM, common practice is pipeline_parallel_size = number of nodes.")
    if p.ray_gpus_per_node is not None and p.ray_gpus_per_node > 0 and p.tensor_parallel_size != p.ray_gpus_per_node:
        warnings.append("For multi-node vLLM, common practice is tensor_parallel_size = GPUs per node.")
    if p.deployment_mode == "spark_ray":
        if not p.spark_master_url:
            errors.append("Spark master URL is required for Spark/Ray export.")
        if not p.spark_executor_instances:
            errors.append("Spark executor instances is required for Spark/Ray export.")
    node_ips = parse_node_ips(p.ray_node_ips)
    if p.deployment_mode in {"ray_cluster", "ray_symmetric_run", "spark_ray"} and not (p.ray_head_address or node_ips):
        warnings.append("Set Ray head address or node IPs before exporting a multi-node script.")
    if p.ray_num_nodes and node_ips and len(node_ips) not in {p.ray_num_nodes, max(p.ray_num_nodes - 1, 0)}:
        warnings.append("Ray node count does not match the node IP list. Include all nodes, or workers only with a head node IP.")
    if p.data_parallel_size and p.data_parallel_size > 1 and p.deployment_mode.startswith("data_parallel"):
        if p.data_parallel_backend != "ray" and not (p.data_parallel_address or p.ray_head_address or node_ips):
            warnings.append("Multi-node data parallel deployments need a data-parallel address or node IP list for per-node commands.")
        if p.data_parallel_size_local is not None and p.data_parallel_size_local > p.data_parallel_size:
            errors.append("Data parallel local size cannot exceed total data parallel size.")
    if p.deployment_mode == "data_parallel_external" and not p.is_moe_model:
        warnings.append("vLLM documents external data-parallel load balancing mainly for MoE deployments. For dense models, consider independent vLLM instances behind a normal load balancer.")
    if p.enable_expert_parallel and not p.is_moe_model:
        warnings.append("Expert parallelism is for MoE models only.")
    if p.speculative_method:
        if p.speculative_method == "draft_model" and not p.speculative_model:
            errors.append("draft_model speculative decoding requires a compatible auxiliary model.")
        if p.speculative_method in {"eagle", "eagle3"}:
            if p.speculative_model:
                warnings.append("EAGLE/EAGLE3 needs a compatible speculator model/head for the target model.")
            else:
                warnings.append("EAGLE/EAGLE3 often uses a compatible speculator model/head; leaving it blank lets vLLM rely on model-native or extra speculative config support when available.")
        if p.speculative_method in {"mtp", "mlp_speculator"} or p.speculative_method.endswith("_mtp"):
            warnings.append("MTP/MLP speculative methods require target-model or checkpoint support. Confirm your model family supports this mode.")
        if p.speculative_method == "custom_class" and not p.speculative_model:
            errors.append("custom_class speculative decoding requires a proposer class path in the model field.")
        if p.speculative_method == "custom_class":
            warnings.append("custom_class speculative decoding is experimental.")
        if p.prompt_lookup_min and p.prompt_lookup_max and p.prompt_lookup_min > p.prompt_lookup_max:
            errors.append("prompt_lookup_min cannot be greater than prompt_lookup_max.")

    return {"errors": errors, "warnings": warnings}


def ensure_profile_valid(p: VllmProfile) -> Dict[str, List[str]]:
    report = validation_report(p)
    if report["errors"]:
        raise HTTPException(status_code=400, detail=report)
    return report


def parse_node_ips(text: str) -> List[str]:
    return [part.strip() for part in text.replace(",", "\n").splitlines() if part.strip()]


def build_speculative_config(p: VllmProfile) -> Optional[Dict[str, Any]]:
    if not p.speculative_method:
        return None
    cfg: Dict[str, Any] = parse_json_object(p.speculative_extra_json, "speculative_extra_json")
    cfg["method"] = p.speculative_method
    if p.speculative_model:
        cfg["model"] = p.speculative_model
    if p.num_speculative_tokens:
        cfg["num_speculative_tokens"] = p.num_speculative_tokens
    if p.draft_tensor_parallel_size:
        cfg["draft_tensor_parallel_size"] = p.draft_tensor_parallel_size
    if p.prompt_lookup_min:
        cfg["prompt_lookup_min"] = p.prompt_lookup_min
    if p.prompt_lookup_max:
        cfg["prompt_lookup_max"] = p.prompt_lookup_max
    return cfg


def build_command(p: VllmProfile) -> List[str]:
    ensure_profile_valid(p)
    if not p.model.strip():
        raise HTTPException(status_code=400, detail="Model is required")
    cmd = ["vllm", "serve", p.model.strip()]

    add_supported_arg(cmd, "--host", p.host)
    add_supported_arg(cmd, "--port", p.port)
    add_supported_arg(cmd, "--api-key", p.api_key)
    add_supported_arg(cmd, "--served-model-name", p.served_model_name)
    add_supported_arg(cmd, "--uvicorn-log-level", p.uvicorn_log_level, omit_values=("info",))
    add_supported_arg(cmd, "--root-path", p.root_path)
    generation_config = p.generation_config_path if p.generation_config == "custom" else p.generation_config
    add_supported_arg(cmd, "--generation-config", generation_config, omit_values=("",))

    add_supported_arg(cmd, "--dtype", p.dtype, omit_values=("auto",))
    add_supported_arg(cmd, "--quantization", p.quantization)
    add_supported_arg(cmd, "--runner", p.runner, omit_values=("auto",))
    add_supported_arg(cmd, "--convert", p.convert, omit_values=("auto",))
    add_supported_arg(cmd, "--task", p.task)
    add_supported_arg(cmd, "--load-format", p.load_format, omit_values=("auto",))
    add_supported_arg(cmd, "--tokenizer-mode", p.tokenizer_mode, omit_values=("auto",))
    add_supported_arg(cmd, "--guided-decoding-backend", p.guided_decoding_backend)
    add_supported_arg(cmd, "--kv-cache-dtype", p.kv_cache_dtype, omit_values=("", "auto"))
    add_supported_arg(cmd, "--max-model-len", p.max_model_len)
    add_supported_arg(cmd, "--gpu-memory-utilization", p.gpu_memory_utilization, omit_values=(None,))
    add_supported_arg(cmd, "--max-num-seqs", p.max_num_seqs)
    add_supported_arg(cmd, "--max-num-batched-tokens", p.max_num_batched_tokens)
    add_supported_arg(cmd, "--swap-space", p.swap_space)
    add_supported_arg(cmd, "--cpu-offload-gb", p.cpu_offload_gb)
    add_supported_arg(cmd, "--download-dir", p.download_dir)
    add_supported_arg(cmd, "--trust-remote-code", p.trust_remote_code, boolean=True)
    add_supported_arg(cmd, "--enforce-eager", p.enforce_eager, boolean=True)
    add_supported_arg(cmd, "--enable-prefix-caching", p.enable_prefix_caching, boolean=True)
    add_supported_arg(cmd, "--prefix-caching-hash-algo", p.prefix_caching_hash_algo)
    add_supported_arg(cmd, "--performance-mode", p.performance_mode)
    add_supported_arg(cmd, "--mamba-cache-dtype", p.mamba_cache_dtype)
    add_supported_arg(cmd, "--mamba-cache-mode", p.mamba_cache_mode)
    add_supported_arg(cmd, "--kv-offloading-backend", p.kv_offloading_backend)
    add_supported_arg(cmd, "--offload-backend", p.offload_backend)
    add_supported_arg(cmd, "--enable-chunked-prefill", p.enable_chunked_prefill, boolean=True)
    add_supported_arg(cmd, "--disable-log-stats", p.disable_log_stats, boolean=True)

    add_supported_arg(cmd, "--tensor-parallel-size", p.tensor_parallel_size, omit_values=(1,))
    add_supported_arg(cmd, "--pipeline-parallel-size", p.pipeline_parallel_size, omit_values=(1,))
    add_supported_arg(cmd, "--distributed-executor-backend", p.distributed_executor_backend)
    add_supported_arg(cmd, "--data-parallel-size", p.data_parallel_size)
    add_supported_arg(cmd, "--data-parallel-size-local", p.data_parallel_size_local)
    add_supported_arg(cmd, "--data-parallel-start-rank", p.data_parallel_start_rank)
    add_supported_arg(cmd, "--data-parallel-address", p.data_parallel_address)
    add_supported_arg(cmd, "--data-parallel-rpc-port", p.data_parallel_rpc_port)
    add_supported_arg(cmd, "--data-parallel-backend", p.data_parallel_backend)
    add_supported_arg(cmd, "--data-parallel-hybrid-lb", p.data_parallel_hybrid_lb, boolean=True)
    add_supported_arg(cmd, "--data-parallel-external-lb", p.data_parallel_external_lb, boolean=True)
    add_supported_arg(cmd, "--data-parallel-rank", p.data_parallel_rank)
    add_supported_arg(cmd, "--data-parallel-multi-port-external-lb", p.data_parallel_multi_port_external_lb, boolean=True)
    add_supported_arg(cmd, "--enable-expert-parallel", p.enable_expert_parallel, boolean=True)
    add_supported_arg(cmd, "--enable-ep-weight-filter", p.enable_ep_weight_filter, boolean=True)
    add_supported_arg(cmd, "--headless", p.headless, boolean=True)
    add_supported_arg(cmd, "--api-server-count", p.api_server_count)
    add_json_arg(cmd, "--structured-outputs-config", p.structured_outputs_config, "structured_outputs_config")
    add_json_arg(cmd, "--compilation-config", p.compilation_config, "compilation_config")
    add_json_arg(cmd, "--kv-transfer-config", p.kv_transfer_config, "kv_transfer_config")
    add_json_arg(cmd, "--additional-config", p.additional_config, "additional_config")
    add_json_arg(cmd, "--override-generation-config", p.override_generation_config, "override_generation_config")

    spec_cfg = build_speculative_config(p)
    if spec_cfg and vllm_options.flag_supported("--speculative-config"):
        cmd.extend(["--speculative-config", json.dumps(spec_cfg, separators=(",", ":"))])

    # Escape hatch for newly added vLLM flags. Admin only; this app should not be exposed publicly.
    if p.advanced_args.strip():
        cmd.extend(shlex.split(p.advanced_args))

    return cmd


def command_for_shell(cmd: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def startup_script(p: VllmProfile) -> str:
    ensure_profile_valid(p)
    env_obj = parse_json_object(p.env_json, "env_json")
    if p.cuda_visible_devices:
        env_obj["CUDA_VISIBLE_DEVICES"] = p.cuda_visible_devices
    env_lines = [f"export {key}={shlex.quote(str(value))}" for key, value in sorted(env_obj.items())]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
    ]
    if env_lines:
        lines.extend(["", *env_lines])
    lines.extend(["", command_for_shell(build_command(p))])
    return "\n".join(lines) + "\n"


def build_env(p: VllmProfile) -> Dict[str, str]:
    ensure_profile_valid(p)
    env = os.environ.copy()
    env_update = parse_json_object(p.env_json, "env_json")
    for k, v in env_update.items():
        if not isinstance(k, str):
            raise HTTPException(status_code=400, detail="env_json keys must be strings")
        env[k] = str(v)
    if p.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = p.cuda_visible_devices
    return env


def process_tree_info() -> Dict[str, Any]:
    proc = state.proc
    if not proc:
        return {}
    info: Dict[str, Any] = {"pid": proc.pid}
    if psutil and state.is_running():
        try:
            pp = psutil.Process(proc.pid)
            info.update({
                "cpu_percent": pp.cpu_percent(interval=0.0),
                "memory_mb": round(pp.memory_info().rss / 1024 / 1024, 1),
                "children": [c.pid for c in pp.children(recursive=True)],
            })
        except Exception:
            pass
    return info


def safe_int(text: str) -> Optional[int]:
    try:
        return int(float(text))
    except Exception:
        return None


def safe_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except Exception:
        return None


def run_command(args: List[str], timeout: int = 5) -> str:
    return subprocess.check_output(args, text=True, timeout=timeout, stderr=subprocess.STDOUT)


def gpu_telemetry() -> Dict[str, Any]:
    gpu_fields = [
        "index",
        "uuid",
        "name",
        "pci.bus_id",
        "driver_version",
        "pstate",
        "memory.used",
        "memory.total",
        "utilization.gpu",
        "utilization.memory",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "clocks.sm",
        "clocks.mem",
        "fan.speed",
    ]
    try:
        out = run_command(
            ["nvidia-smi", f"--query-gpu={','.join(gpu_fields)}", "--format=csv,noheader,nounits"],
            timeout=6,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "gpus": [], "processes": [], "summary": {}}

    gpus: List[Dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(gpu_fields):
            continue
        row = dict(zip(gpu_fields, parts))
        used = safe_int(row["memory.used"])
        total = safe_int(row["memory.total"])
        power = safe_float(row["power.draw"].replace("N/A", ""))
        power_limit = safe_float(row["power.limit"].replace("N/A", ""))
        gpu = {
            "index": safe_int(row["index"]),
            "uuid": row["uuid"],
            "name": row["name"],
            "pci_bus_id": row["pci.bus_id"],
            "driver_version": row["driver_version"],
            "pstate": row["pstate"],
            "memory_used_mb": used,
            "memory_total_mb": total,
            "memory_percent": round((used / total) * 100, 1) if used is not None and total else None,
            "gpu_utilization_percent": safe_int(row["utilization.gpu"]),
            "memory_utilization_percent": safe_int(row["utilization.memory"]),
            "temperature_c": safe_int(row["temperature.gpu"]),
            "power_draw_w": power,
            "power_limit_w": power_limit,
            "power_percent": round((power / power_limit) * 100, 1) if power is not None and power_limit else None,
            "sm_clock_mhz": safe_int(row["clocks.sm"]),
            "memory_clock_mhz": safe_int(row["clocks.mem"]),
            "fan_percent": safe_int(row["fan.speed"].replace("N/A", "")),
        }
        gpus.append(gpu)

    processes: List[Dict[str, Any]] = []
    try:
        proc_out = run_command(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            timeout=6,
        )
        for line in proc_out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                processes.append({
                    "gpu_uuid": parts[0],
                    "pid": safe_int(parts[1]),
                    "process_name": parts[2],
                    "used_memory_mb": safe_int(parts[3]),
                })
    except Exception:
        pass

    total_mem = sum(g["memory_total_mb"] or 0 for g in gpus)
    used_mem = sum(g["memory_used_mb"] or 0 for g in gpus)
    summary = {
        "gpu_count": len(gpus),
        "memory_used_mb": used_mem,
        "memory_total_mb": total_mem,
        "memory_percent": round((used_mem / total_mem) * 100, 1) if total_mem else None,
        "max_gpu_utilization_percent": max([g["gpu_utilization_percent"] or 0 for g in gpus], default=None),
        "max_temperature_c": max([g["temperature_c"] or 0 for g in gpus], default=None),
    }
    return {"ok": True, "gpus": gpus, "processes": processes, "summary": summary}


def ray_head(p: VllmProfile) -> str:
    ips = parse_node_ips(p.ray_node_ips)
    return p.ray_head_address or (ips[0] if ips else "HEAD_IP")


def export_ray_script(p: VllmProfile) -> str:
    ensure_profile_valid(p)
    cmd = command_for_shell(build_command(p))
    head = ray_head(p)
    port = p.ray_port
    nodes = parse_node_ips(p.ray_node_ips)
    workers = worker_ips_for_runbook(p) or ["WORKER_IP_1", "WORKER_IP_2"]
    node_list = " ".join(shlex.quote(node) for node in nodes) if nodes else "HEAD_IP WORKER_IP_1 WORKER_IP_2"
    min_nodes = p.ray_min_nodes or p.ray_num_nodes or max(len(nodes), 1)
    gpus_per_node = p.ray_gpus_per_node if p.ray_gpus_per_node is not None else p.tensor_parallel_size

    if p.deployment_mode == "ray_symmetric_run":
        return f"""#!/usr/bin/env bash
set -euo pipefail

# Run this command from each provisioned node with a shared environment.
# The head address must be reachable from all workers.
ray symmetric-run \\
  --address {shlex.quote(head)}:{port} \\
  --min-nodes {min_nodes} \\
  --num-gpus {gpus_per_node} \\
  -- {cmd}
"""

    worker_lines = "\n".join(
        f"ssh {shlex.quote(worker)} 'ray stop --force || true; ray start --address={shlex.quote(head)}:{port}'"
        for worker in workers
    )
    return f"""#!/usr/bin/env bash
set -euo pipefail

# Nodes: {node_list}
# Run from the head/API node after passwordless SSH and identical Python/vLLM environments are ready.
ray stop --force || true
ray start --head --node-ip-address={shlex.quote(head)} --port={port} --dashboard-host=0.0.0.0

{worker_lines}

# Start vLLM on the head/API node. For multi-node models, use --distributed-executor-backend ray.
{cmd}
"""


def export_spark_ray_script(p: VllmProfile) -> str:
    ensure_profile_valid(p)
    cmd = command_for_shell(build_command(p))
    spark_conf = parse_json_object(p.spark_conf_json, "spark_conf_json")
    spark_args = shlex.split(p.spark_submit_args or "")
    instances = p.spark_executor_instances or p.ray_num_nodes or 2
    cores = p.spark_executor_cores or 4
    gpus = p.spark_executor_gpus if p.spark_executor_gpus is not None else (p.ray_gpus_per_node or p.tensor_parallel_size)
    master = p.spark_master_url or "spark://SPARK_MASTER:7077"
    app_name = p.spark_app_name or "vllm-ray-cluster"
    head = ray_head(p)
    port = p.ray_port
    min_nodes = p.ray_min_nodes or p.ray_num_nodes or instances
    gpus_per_node = p.ray_gpus_per_node if p.ray_gpus_per_node is not None else gpus
    submit_lines = [
        "spark-submit",
        f"  --master {shlex.quote(master)}",
        f"  --name {shlex.quote(app_name)}",
        f"  --conf spark.executor.instances={instances}",
        f"  --conf spark.executor.cores={cores}",
        f"  --conf spark.executor.resource.gpu.amount={gpus}",
    ]
    submit_lines.extend(f"  --conf {shlex.quote(str(k))}={shlex.quote(str(v))}" for k, v in spark_conf.items())
    if spark_args:
        submit_lines.append("  " + " ".join(shlex.quote(arg) for arg in spark_args))
    submit_lines.extend([
        '  "$SPARK_VLLM_DRIVER"',
        f"  --ray-head-address {shlex.quote(head)}",
        f"  --ray-port {port}",
        f"  --ray-min-nodes {min_nodes}",
        f"  --ray-gpus-per-node {gpus_per_node}",
        f"  --vllm-command {shlex.quote(cmd)}",
    ])
    submit_script = " \\\n".join(submit_lines)

    return f"""#!/usr/bin/env bash
set -euo pipefail

# This exports a Spark-launched Ray/vLLM job for clusters where Spark owns allocation.
# Provide a Python driver at $SPARK_VLLM_DRIVER that starts Ray workers on executors
# and runs the command below on the Ray head/API node.
: "${{SPARK_VLLM_DRIVER:=spark_vllm_ray_driver.py}}"

{submit_script}
"""


def imported_profiles(payload: Dict[str, Any]) -> List[VllmProfile]:
    raw_profiles: Any
    if isinstance(payload.get("profiles"), list):
        raw_profiles = payload["profiles"]
    elif isinstance(payload.get("profile"), dict):
        raw_profiles = [payload["profile"]]
    elif "model" in payload:
        raw_profiles = [payload]
    else:
        raise HTTPException(status_code=400, detail="Import JSON must contain profiles, profile, or a single profile object.")
    profiles = [VllmProfile(**item) for item in raw_profiles]
    for profile in profiles:
        ensure_profile_valid(profile)
    return profiles


def worker_ips_for_runbook(p: VllmProfile) -> List[str]:
    ips = parse_node_ips(p.ray_node_ips)
    if ips and p.ray_head_address and ips[0] == p.ray_head_address:
        return ips[1:]
    if ips and not p.ray_head_address and len(ips) > 1:
        return ips[1:]
    return ips


def data_parallel_node_commands(p: VllmProfile) -> List[str]:
    if not p.data_parallel_size or p.data_parallel_size <= 1:
        return []
    if p.data_parallel_backend == "ray":
        return [
            "# Ray data-parallel backend starts local and remote DP ranks from one vLLM command.",
            command_for_shell(build_command(p)),
        ]

    head = p.data_parallel_address or ray_head(p)
    workers = worker_ips_for_runbook(p)
    node_names = [head] + workers
    if len(node_names) == 1 and p.ray_num_nodes and p.ray_num_nodes > 1:
        node_names.extend(f"WORKER_IP_{idx}" for idx in range(1, p.ray_num_nodes))
    if len(node_names) == 1:
        return [command_for_shell(build_command(p))]

    port = p.data_parallel_rpc_port or 13345
    total = p.data_parallel_size
    commands: List[str] = []
    start_rank = p.data_parallel_start_rank or 0

    for index, node in enumerate(node_names):
        remaining = total - start_rank
        if remaining <= 0:
            break
        remaining_nodes = len(node_names) - index
        if index == 0 and p.data_parallel_size_local is not None:
            local_size = min(p.data_parallel_size_local, remaining)
        else:
            local_size = max(1, (remaining + remaining_nodes - 1) // remaining_nodes)
            local_size = min(local_size, remaining)
        node_profile = VllmProfile(**p.model_dump())
        node_profile.data_parallel_address = head
        node_profile.data_parallel_rpc_port = port
        node_profile.data_parallel_start_rank = start_rank if index > 0 else p.data_parallel_start_rank
        node_profile.data_parallel_size_local = local_size
        node_profile.data_parallel_rank = None
        node_profile.headless = index > 0
        node_profile.host = p.host if index == 0 else "127.0.0.1"
        commands.append(f"# Node {index} ({node})")
        commands.append(command_for_shell(build_command(node_profile)))
        start_rank += local_size

    if start_rank < total:
        commands.append(f"# Add more nodes or increase data_parallel_size_local; only {start_rank} of {total} DP ranks were assigned.")
    return commands


def cluster_runbook(p: VllmProfile) -> Dict[str, Any]:
    report = validation_report(p)
    head = ray_head(p)
    workers = worker_ips_for_runbook(p) or ["WORKER_IP_1"]
    node_count = p.ray_num_nodes or (1 + len(workers))
    gpus_per_node = p.ray_gpus_per_node or p.tensor_parallel_size or 1
    serve_profile = VllmProfile(**p.model_dump())

    if p.deployment_mode in {"ray_cluster", "ray_symmetric_run"}:
        serve_profile.distributed_executor_backend = "ray"
        serve_profile.tensor_parallel_size = gpus_per_node
        serve_profile.pipeline_parallel_size = node_count
    if p.deployment_mode == "data_parallel_hybrid":
        serve_profile.data_parallel_hybrid_lb = True
    if p.deployment_mode == "data_parallel_external":
        serve_profile.data_parallel_external_lb = True
    if p.deployment_mode == "expert_parallel_moe":
        serve_profile.is_moe_model = True
        serve_profile.enable_expert_parallel = True
        serve_profile.enable_ep_weight_filter = True

    serve_cmd = command_for_shell(build_command(serve_profile))
    dp_node_cmds = data_parallel_node_commands(serve_profile)
    worker_cmds = [
        f"ssh {shlex.quote(worker)} 'export VLLM_HOST_IP={shlex.quote(worker)}; ray start --address={shlex.quote(head)}:{p.ray_port}'"
        for worker in workers
    ]
    lb_cmds: List[str] = []
    if p.external_load_balancer or p.deployment_mode in {"data_parallel_external", "data_parallel_hybrid"}:
        lb_cmds = [
            "# Put Nginx, HAProxy, Envoy, or your platform load balancer in front of the vLLM API nodes.",
            "# Health check: GET /health on each API node.",
        ]
    if p.deployment_mode == "data_parallel_external" and p.data_parallel_size:
        lb_cmds.append("# Generate one vLLM command per rank/node with --data-parallel-rank and distinct ports as needed.")

    return {
        "ok": not report["errors"],
        **report,
        "sections": {
            "A. Head node commands": [
                f"export VLLM_HOST_IP={head}",
                f"ray start --head --node-ip-address={head} --port={p.ray_port} --dashboard-host=0.0.0.0",
            ] if p.deployment_mode in {"ray_cluster", "ray_symmetric_run", "expert_parallel_moe"} else ["# Run vLLM locally on this host."],
            "B. Worker node commands": worker_cmds if p.deployment_mode in {"ray_cluster", "ray_symmetric_run", "expert_parallel_moe"} else ["# No worker nodes required for this mode."],
            "C. vLLM serve command": [serve_cmd],
            "D. Per-node data-parallel commands": dp_node_cmds or ["# Not required for this profile."],
            "E. Optional load balancer commands": lb_cmds or ["# Not required for this profile."],
            "F. Verification commands": ["ray status", "ray list nodes", "nvidia-smi", f"curl http://{p.host}:{p.port}/health"],
            "G. Stop/cleanup commands": ["pkill -f 'vllm serve' || true", "ray stop --force"],
            "H. Security notes": [
                "Keep Ray and vLLM distributed traffic on a private trusted network.",
                "Do not expose vLLM Web publicly.",
                "Treat stored API keys, Hugging Face tokens, and profile exports as sensitive.",
            ],
        },
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/api/profiles")
def api_profiles() -> Dict[str, Any]:
    return {"profiles": [p.model_dump() for p in load_profiles()]}


@app.get("/api/vllm/options")
def api_vllm_options() -> Dict[str, Any]:
    return vllm_options.payload()


@app.post("/api/profiles/validate")
def api_validate_profile(payload: ProfileValidationRequest) -> Dict[str, Any]:
    report = validation_report(payload.profile)
    return {"ok": not report["errors"], **report}


@app.get("/api/profiles/export")
def api_export_profiles() -> Dict[str, Any]:
    return profile_export_payload(load_profiles())


@app.get("/api/profiles/{profile_id}/export")
def api_export_profile(profile_id: str) -> Dict[str, Any]:
    return profile_export_payload([get_profile(profile_id)])


@app.post("/api/profiles/import")
def api_import_profiles(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    profiles = load_profiles()
    incoming = imported_profiles(payload)
    replace_existing = bool(payload.get("replace_existing", True))
    regenerate_ids = bool(payload.get("regenerate_ids", False))
    existing_by_id = {p.id: idx for idx, p in enumerate(profiles)}
    imported: List[Dict[str, Any]] = []

    for profile in incoming:
        if regenerate_ids or not profile.id:
            profile.id = str(uuid.uuid4())
        if replace_existing and profile.id in existing_by_id:
            profiles[existing_by_id[profile.id]] = profile
        else:
            if profile.id in existing_by_id:
                profile.id = str(uuid.uuid4())
            profiles.append(profile)
        imported.append(profile.model_dump())

    save_profiles(profiles)
    return {"ok": True, "imported": imported, "count": len(imported)}


@app.post("/api/profiles")
def api_save_profile(payload: ProfilePatch) -> Dict[str, Any]:
    profiles = load_profiles()
    profile = payload.profile
    report = ensure_profile_valid(profile)
    if not profile.id:
        profile.id = str(uuid.uuid4())
    replaced = False
    for idx, item in enumerate(profiles):
        if item.id == profile.id:
            profiles[idx] = profile
            replaced = True
            break
    if not replaced:
        profiles.append(profile)
    save_profiles(profiles)
    return {"ok": True, "profile": profile.model_dump(), **report}


@app.post("/api/profiles/{profile_id}/clone")
def api_clone_profile(profile_id: str) -> Dict[str, Any]:
    p = get_profile(profile_id)
    clone = VllmProfile(**p.model_dump())
    clone.id = str(uuid.uuid4())
    clone.name = f"{p.name} copy"
    profiles = load_profiles()
    profiles.append(clone)
    save_profiles(profiles)
    return {"ok": True, "profile": clone.model_dump()}


@app.delete("/api/profiles/{profile_id}")
def api_delete_profile(profile_id: str) -> Dict[str, Any]:
    if state.is_running() and state.profile_id == profile_id:
        raise HTTPException(status_code=409, detail="Stop the running server before deleting this profile")
    profiles = [p for p in load_profiles() if p.id != profile_id]
    if not profiles:
        profiles = [VllmProfile()]
    save_profiles(profiles)
    return {"ok": True}


@app.get("/api/profiles/{profile_id}/command")
def api_command(profile_id: str) -> Dict[str, Any]:
    p = get_profile(profile_id)
    cmd = build_command(p)
    return {"cmd": cmd, "shell": command_for_shell(cmd), **validation_report(p)}


@app.post("/api/profiles/command")
def api_preview_command(payload: ProfileValidationRequest) -> Dict[str, Any]:
    report = validation_report(payload.profile)
    if report["errors"]:
        return {"ok": False, "cmd": [], "shell": "", "speculative_json": None, **report}
    cmd = build_command(payload.profile)
    spec_cfg = build_speculative_config(payload.profile)
    return {
        "ok": True,
        "cmd": cmd,
        "shell": command_for_shell(cmd),
        "speculative_json": spec_cfg,
        **report,
    }


@app.post("/api/start-script")
def api_start_script(payload: ProfileValidationRequest) -> PlainTextResponse:
    return PlainTextResponse(startup_script(payload.profile))


@app.post("/api/cluster/runbook")
def api_cluster_runbook(payload: ProfileValidationRequest) -> Dict[str, Any]:
    return cluster_runbook(payload.profile)


def start_profile(p: VllmProfile, profile_id: Optional[str] = None) -> Dict[str, Any]:
    with state._lock:
        if state.is_running():
            raise HTTPException(status_code=409, detail="A vLLM server is already running. Stop it first.")
        cmd = build_command(p)
        env = build_env(p)
        state.log_lines.clear()
        state.log_file = LOG_DIR / f"vllm-{time.strftime('%Y%m%d-%H%M%S')}.log"
        state.append_log("Starting vLLM")
        state.append_log(command_for_shell(cmd))
        try:
            kwargs: Dict[str, Any] = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "bufsize": 1,
                "env": env,
            }
            if platform.system().lower() != "windows":
                kwargs["start_new_session"] = True
            state.proc = subprocess.Popen(cmd, **kwargs)
        except FileNotFoundError:
            state.proc = None
            raise HTTPException(status_code=500, detail="vllm command not found. Activate your vLLM environment before starting this UI.")
        except Exception as exc:
            state.proc = None
            raise HTTPException(status_code=500, detail=str(exc))
        state.profile_id = profile_id or p.id
        state.command = cmd
        state.started_at = time.time()
        state.start_reader()
    return {"ok": True, "pid": state.proc.pid if state.proc else None, "command": command_for_shell(cmd)}


@app.post("/api/profiles/{profile_id}/start")
def api_start(profile_id: str) -> Dict[str, Any]:
    return start_profile(get_profile(profile_id), profile_id)


@app.post("/api/start")
def api_start_unsaved(payload: ProfileValidationRequest) -> Dict[str, Any]:
    return start_profile(payload.profile, payload.profile.id)


@app.post("/api/stop")
def api_stop() -> Dict[str, Any]:
    with state._lock:
        if not state.proc:
            return {"ok": True, "message": "No process"}
        proc = state.proc
        if proc.poll() is None:
            state.append_log("Stopping vLLM")
            try:
                if platform.system().lower() == "windows":
                    proc.terminate()
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    state.append_log("SIGTERM timeout; killing vLLM")
                    if platform.system().lower() == "windows":
                        proc.kill()
                    else:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc))
        state.append_log("vLLM stopped")
        state.proc = None
        state.profile_id = None
        state.command = []
        state.started_at = None
    return {"ok": True}


@app.post("/api/profiles/{profile_id}/restart")
def api_restart(profile_id: str) -> Dict[str, Any]:
    if state.is_running():
        api_stop()
    return api_start(profile_id)


@app.get("/api/status")
def api_status(request: Request) -> Dict[str, Any]:
    running = state.is_running()
    exit_code = None
    if state.proc and not running:
        exit_code = state.proc.poll()
    uptime = round(time.time() - state.started_at, 1) if running and state.started_at else None
    dashboard_host = request.url.hostname or ""
    return {
        "running": running,
        "exit_code": exit_code,
        "profile_id": state.profile_id,
        "command": command_for_shell(state.command) if state.command else None,
        "started_at": state.started_at,
        "uptime_seconds": uptime,
        "process": process_tree_info(),
        "log_file": str(state.log_file) if state.log_file else None,
        "dashboard": {
            "host": dashboard_host,
            "is_local": dashboard_host in LOCAL_HOSTS,
            "warning": None if dashboard_host in LOCAL_HOSTS else "vLLM Web is not being accessed through localhost. Protect this service with auth and a private network.",
        },
    }


@app.get("/api/logs")
def api_logs() -> PlainTextResponse:
    return PlainTextResponse("\n".join(state.log_lines))


@app.get("/api/gpu")
def api_gpu() -> Dict[str, Any]:
    return gpu_telemetry()


@app.post("/api/chat")
def api_chat(req: ChatRequest) -> Dict[str, Any]:
    if not state.profile_id:
        raise HTTPException(status_code=409, detail="No active profile")
    p = get_profile(state.profile_id)
    model_name = req.model or p.served_model_name or p.model
    url = f"http://{p.host}:{p.port}/v1/chat/completions"
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if p.api_key:
        headers["Authorization"] = f"Bearer {p.api_key}"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        return {"ok": r.ok, "status_code": r.status_code, "response": r.json() if r.text else None}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/export/systemd/{profile_id}")
def api_export_systemd(profile_id: str) -> PlainTextResponse:
    p = get_profile(profile_id)
    cmd = command_for_shell(build_command(p))
    env_lines = []
    env_obj = parse_json_object(p.env_json, "env_json")
    if p.cuda_visible_devices:
        env_obj["CUDA_VISIBLE_DEVICES"] = p.cuda_visible_devices
    for k, v in env_obj.items():
        assignment = f"{k}={v}"
        env_lines.append(f"Environment={shlex.quote(assignment)}")
    body = f"""[Unit]\nDescription=vLLM server - {p.name}\nAfter=network-online.target\nWants=network-online.target\n\n[Service]\nType=simple\nWorkingDirectory={shlex.quote(str(ROOT))}\n{chr(10).join(env_lines)}\nExecStart={cmd}\nRestart=on-failure\nRestartSec=10\n\n[Install]\nWantedBy=multi-user.target\n"""
    return PlainTextResponse(body)


@app.get("/api/export/ray/{profile_id}")
def api_export_ray(profile_id: str) -> PlainTextResponse:
    p = get_profile(profile_id)
    return PlainTextResponse(export_ray_script(p))


@app.get("/api/export/spark/{profile_id}")
def api_export_spark(profile_id: str) -> PlainTextResponse:
    p = get_profile(profile_id)
    return PlainTextResponse(export_spark_ray_script(p))
