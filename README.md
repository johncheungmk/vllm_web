# vLLM Web

A small local web app for configuring and launching `vllm serve` profiles without editing startup scripts every time.

It is designed for an admin workstation or an inference server, not for public Internet exposure. It can start and stop local processes and may store secrets in `data/profiles.json`.

## Quick start

```bash
git clone <your-repo-url>
cd vllm-ui-dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8899
```

Open:

```text
http://127.0.0.1:8899
```

Use the same Python environment where `vllm` works. If `vllm` is not installed in that environment, vLLM Web still opens and uses fallback option lists, but the **Start** action cannot launch a real server until `vllm` is available.

## What it does

- Save multiple vLLM launch profiles.
- Generate a live `vllm serve ...` command preview from the current UI fields, including unsaved edits.
- Use dropdowns for fixed or semi-fixed vLLM values such as dtype, quantization, runner, convert, load format, log level, generation config, distributed executor backend, guided decoding backend, and speculative decoding method.
- Detect installed `vllm serve --help` options when available and hide or omit unsupported flags.
- Switch between Wizard mode for step-by-step setup and Expert mode for the full configuration surface.
- Start, stop, and restart the vLLM server process.
- Show logs from the running process.
- Show richer `nvidia-smi` GPU telemetry: memory, utilization, temperature, power, clocks, P-state, and compute processes.
- Test the OpenAI-compatible `/v1/chat/completions` endpoint.
- Configure distributed inference options and generate cluster runbooks: local, single-node multi-GPU, Ray multi-node, data parallel, expert parallel MoE, and Spark launcher helpers.
- Configure deployment export metadata for local, Ray cluster, Ray symmetric-run, and Spark-managed Ray deployments.
- Configure speculative decoding using the current `--speculative-config` JSON style.
- Import and export the selected profile as JSON.
- Export example systemd service, Ray launch script, and Spark/Ray launch script.

## Important limitation

Most vLLM engine parameters are startup parameters. Changing the model, tensor parallel size, pipeline parallel size, quantization, KV cache settings, speculative decoding, or data parallel topology requires restarting vLLM. vLLM Web manages this restart flow; it does not hot-edit a live vLLM engine.

vLLM Web validates profile shape before saving or starting: JSON fields must parse as objects, ports and sizes must be in range, advanced arguments must parse with `shlex.split`, and unsafe settings are shown as warnings. Warnings do not block startup; errors do.

Blank optional text fields are stored as empty strings. This keeps profile import/export stable and avoids `null` values causing Pydantic validation errors for textareas such as Ray node IPs, Spark submit args, advanced args, and notes.

## Save, Start, and Export

**Save Profile** persists the selected profile to `data/profiles.json`.

**Start** runs the current visible configuration, including unsaved edits. If fields have changed since the last save, vLLM Web shows an **Unsaved changes** badge, but Start is not blocked.

**vLLM server: Stopped** means vLLM Web is running but no `vllm serve` process is currently active. After starting vLLM from the UI, the header changes to show the running process state.

**Download startup script** exports a `start-vllm.sh` script from the current wizard settings. Expert mode also provides systemd, Ray, and Spark/Ray export tools in the collapsed **Scripts and profile tools** section.

## Install and run

Use the same Python environment where `vllm` works, or activate that environment before starting vLLM Web.

```bash
cd vllm-ui-dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8899
```

Open:

```text
http://127.0.0.1:8899
```

Or:

```bash
./scripts/run_ui.sh
```

## Basic vLLM profile example

- Model: `Qwen/Qwen3-8B`
- Host: `127.0.0.1`
- Port: `8000`
- dtype: `auto`
- Tensor parallel size: `1`
- Pipeline parallel size: `1`

The generated command may look like:

```bash
vllm serve Qwen/Qwen3-8B --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.9
```

## Command preview

The command preview updates automatically whenever you change a field. You do not need to save first. This is useful for checking model, dtype, quantization, host, port, parallel sizes, speculative decoding, and advanced args before committing the profile.

In Wizard mode, use **Download startup script** on the Review step to save a runnable shell script. There is no refresh button because preview is dynamic.

Validation messages are shown in vLLM Web instead of raw JSON alert popups. Numeric fields use browser constraints where possible: ports are `1` to `65535`, parallel sizes are positive integers, and GPU memory utilization is `0.01` to `1`.

vLLM Web calls `/api/vllm/options`, which tries to inspect the installed `vllm serve --help`. If vLLM is not installed in the app environment, fallback choices are used and the UI shows a warning.

## Wizard and Expert Mode

Wizard mode guides setup through Basic, Performance, Distributed, Speculative decoding, and Review steps. The final Review step can either output a startup script for manual use or start vLLM directly from vLLM Web using the current unsaved form values.

Expert mode shows the full tabbed configuration surface and export/start controls for users who want to change any option directly.

Wizard mode includes common fields and presets:

- Single GPU
- Single node multi-GPU
- Ray multi-node
- MoE expert parallel
- Low latency speculative decoding
- High throughput serving
- Memory saving serving

Expert mode reveals compatibility and advanced fields, including custom CLI args and JSON configs for structured outputs, compilation, KV transfer, additional config, and override generation config. Expert args may override or conflict with UI fields.

## Dropdown fields

The UI uses dropdowns for values that are fixed or mostly fixed in vLLM:

- `dtype`: `auto`, `half`, `float16`, `bfloat16`, `float`, `float32`
- `quantization`: common backends including `awq`, `gptq`, `bitsandbytes`, `fp8`, `gguf`, Marlin variants, `modelopt`, `nvfp4`, `inc`, and `fbgemm_fp8`
- `runner`: `auto`, `generate`, `pooling`, `draft`
- `convert`: `auto`, `none`, `embed`, `classify`
- legacy `task`: available only in Advanced compatibility mode when supported
- `load_format`: `auto`, `pt`, `safetensors`, `npcache`, `dummy`, `tensorizer`, `bitsandbytes`, `sharded_state`, `gguf`, `mistral`, `runai_streamer`, `fastsafetensors`
- `distributed_executor_backend`: auto, `mp`, `ray`, `external_launcher`, `uni`
- `uvicorn_log_level`: `critical`, `error`, `warning`, `info`, `debug`, `trace`
- `generation_config`: default, `auto`, `vllm`, custom path
- `guided_decoding_backend`: default, `auto`, `xgrammar`, `outlines`, `guidance`
- `speculative_method`: none, `draft_model`, `eagle`, `eagle3`, `mtp`, `ngram`, `ngram_gpu`, `suffix`, `medusa`, `mlp_speculator`, `custom_class`, plus detected installed-version methods

Several semi-fixed fields include a custom option for newer vLLM versions. The expert-only Advanced vLLM args field remains free text and is parsed with `shlex.split` before save/start.

## 4-GPU single-node tensor parallel example

Set:

- Tensor parallel size: `4`
- Pipeline parallel size: `1`
- Distributed executor backend: `mp` or leave blank for vLLM default
- CUDA_VISIBLE_DEVICES: `0,1,2,3`

## Multi-node Ray example

Current vLLM guidance is:

- Single-node multi-GPU: use tensor parallelism.
- Multi-node: use Ray and usually set tensor parallel size to GPUs per node and pipeline parallel size to number of nodes.
- Ensure all nodes have the same Python, CUDA, PyTorch, vLLM, and model path/cache.

1. On the head node:

```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0
```

2. On each worker node:

```bash
ray start --address=HEAD_IP:6379
```

3. In the vLLM Web profile:

- Distributed executor backend: `ray`
- Tensor parallel size: GPUs per node, for example `8`
- Pipeline parallel size: number of nodes, for example `2`
- Deployment mode: `ray_cluster`
- Ray head address: the head node IP
- Ray node IPs: one node IP per line, with the head first

Then preview the command or export the Ray script. The UI itself starts only a process on the current host; use the exported script when you want vLLM Web to produce repeatable cluster launch commands.

The Distributed tab also generates a **Cluster Runbook** with:

- Head node commands
- Worker node commands
- Final `vllm serve` command
- Optional load balancer notes
- Verification commands such as `ray status`, `ray list nodes`, and `nvidia-smi`
- Stop/cleanup commands
- Security notes

For Ray symmetric-run clusters, set deployment mode to `ray_symmetric_run`. The export uses:

```bash
ray symmetric-run --address HEAD_IP:6379 --min-nodes 2 --num-gpus 8 -- vllm serve MODEL ...
```

This is useful on HPC-style environments where the same command is launched on every allocated node.

## Spark-managed Ray deployment

Spark is treated as the resource allocator and job launcher; vLLM still uses Ray for multi-node execution. Set:

- Deployment mode: `spark_ray`
- Distributed executor backend: `ray`
- Spark master URL: for example `spark://spark-master:7077`, `yarn`, or your cluster manager URL
- Spark executor instances: number of Spark executors/nodes to allocate
- Spark executor GPUs: GPUs per executor/node
- Ray head address and Ray port

The Spark export emits a `spark-submit` wrapper and passes the generated `vllm serve ...` command to a driver script named by `$SPARK_VLLM_DRIVER`:

```bash
SPARK_VLLM_DRIVER=/opt/vllm/spark_vllm_ray_driver.py ./exported-spark-script.sh
```

That driver is environment-specific because Spark clusters differ in how executors expose host IPs, GPUs, and worker lifecycle hooks. Use it as the integration point for your existing Spark cluster bootstrap.

Spark mode is marked experimental because Spark is external orchestration, not core vLLM serving.

## Data-parallel deployment example

For internal vLLM data parallel load balancing on one 8-GPU node:

- Data parallel size: `4`
- Tensor parallel size: `2`

This generates the equivalent of:

```bash
vllm serve MODEL --data-parallel-size 4 --tensor-parallel-size 2
```

For multi-node data parallel, use the `data_parallel_address`, `data_parallel_size_local`, `data_parallel_start_rank`, and `headless` fields to build the per-node commands.

External data-parallel load balancing is mainly documented for MoE deployments. For dense models, vLLM Web warns and suggests independent vLLM instances behind a normal load balancer.

## Expert Parallel MoE

For MoE models, choose **Expert parallel MoE** and enable expert parallel fields. When supported by installed vLLM, vLLM Web can include:

```text
--enable-expert-parallel
--enable-ep-weight-filter
```

Unsupported installed-version flags are omitted from generated commands and shown as warnings.

## Speculative decoding examples

The speculative decoding tab is method-aware. If the method is `none`, all speculative fields are hidden and `--speculative-config` is omitted.

### Draft model

- Enable speculative decoding: checked
- Method: `draft_model`
- Draft / auxiliary model: `Qwen/Qwen3-0.6B`
- Num speculative tokens: `5`

Generated part:

```bash
--speculative-config '{"method":"draft_model","model":"Qwen/Qwen3-0.6B","num_speculative_tokens":5}'
```

### N-gram

- Method: `ngram`
- Num speculative tokens: `4`
- prompt lookup min/max:

```json
{"prompt_lookup_min": 2, "prompt_lookup_max": 5}
```

N-gram and suffix decoding do not require a separate draft model.

### EAGLE / EAGLE3

EAGLE and EAGLE3 require a compatible speculator model or head. vLLM Web warns when this is missing because using the target model alone is usually wrong.

### Suffix

Suffix decoding does not require a separate draft model. Use Extra speculative JSON for installed-version-specific suffix fields.

### MTP

Use MTP methods when the target model or checkpoint supports MTP-style speculative decoding. Some installed vLLM versions may expose model-specific methods such as DeepSeek or Qwen MTP variants; detected methods are added to the dropdown when found in `vllm serve --help`.

## Use with Codex

From the folder where you want the project:

```bash
codex
```

Paste this task:

```text
Build and improve vLLM Web in this repository. Keep it local-first and safe by default.

Requirements:
1. Backend: FastAPI. Keep profiles in data/profiles.json unless I ask for SQLite.
2. Frontend: simple web app; no heavy framework unless necessary.
3. Must support these config sections:
   - Basic: model, served model name, host, port, API key, dtype, quantization, trust_remote_code, generation_config.
   - Performance: max_model_len, gpu_memory_utilization, max_num_seqs, max_num_batched_tokens, kv_cache_dtype, swap_space, cpu_offload_gb, prefix caching, chunked prefill, enforce eager.
   - Distributed: tensor_parallel_size, pipeline_parallel_size, distributed_executor_backend, data_parallel_size, data_parallel_size_local, data_parallel_start_rank, data_parallel_address, data_parallel_rpc_port, data_parallel_backend, data_parallel_hybrid_lb, headless, api_server_count.
   - Speculative decoding: --speculative-config JSON with method, model, num_speculative_tokens, draft_tensor_parallel_size, and extra JSON.
   - Advanced: CUDA_VISIBLE_DEVICES, environment JSON, advanced vLLM args.
4. The UI should preview the exact shell command before starting.
5. Start/stop/restart should manage the vLLM subprocess and show logs.
6. Add GPU status from nvidia-smi when available.
7. Add a test chat request against /v1/chat/completions.
8. Add validation: JSON fields must be valid JSON; dangerous public binding should show a warning if vLLM Web is not localhost.
9. Do not execute arbitrary shell strings except the generated argv list; advanced args must be parsed with shlex.split.
10. Keep README updated with examples for single GPU, tensor parallel, pipeline parallel, Ray multi-node, data parallel, and speculative decoding.
```

## Production hardening checklist

- Bind vLLM Web to `127.0.0.1` and access through SSH tunnel, VPN, or a protected reverse proxy.
- Do not store long-term Hugging Face tokens or API keys in profiles unless the host is trusted.
- Use a service account with limited permissions.
- For multi-node vLLM, isolate the cluster network; vLLM's internal distributed channels are not encrypted by default.
- Keep distributed vLLM and Ray traffic on a private trusted network.
- Use systemd or a supervisor if the vLLM process must survive vLLM Web restarts.
- For public model API traffic, put Nginx/Caddy/Traefik in front for TLS, auth, rate limits, logging, and access control.
- Keep imported/exported profile JSON in a secure location; it can contain API keys and environment variables.
- Review warnings in the UI before starting a profile, especially public bind addresses, Ray topology mismatches, and `trust_remote_code`.

## Files

```text
app/main.py             FastAPI backend and vLLM process manager
app/static/index.html   vLLM Web interface
app/static/app.js       Frontend logic
app/static/style.css    Styles
scripts/run_ui.sh       Quick local launcher
scripts/install_systemd.sh Optional vLLM Web service installer
```
