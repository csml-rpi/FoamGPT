# utils.py
import re
import subprocess
import os
import signal
import threading
import json
from typing import Optional, Any, Type, TypedDict, List, Dict, Union
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
import tiktoken
from langchain_aws import ChatBedrock, ChatBedrockConverse
from langchain_anthropic import ChatAnthropic
from pathlib import Path
import tracking_aws
import requests
import time
import random
from botocore.exceptions import ClientError
import shutil
from config import Config
from langchain_ollama import ChatOllama
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    HuggingFaceEmbeddings = None


# Global dictionary to store loaded FAISS databases
FAISS_DB_CACHE = {}

def get_embedding_model(config: Optional[Config] = None):
    """Return an embedding model based on the provided config.

    Note: historically this module accessed Config.* class attributes at import time.
    That works for defaults but breaks when callers pass a customized Config instance.
    """
    cfg = config or Config()

    provider = (cfg.embedding_provider or "openai").lower()
    model = cfg.embedding_model

    if provider == "openai":
        return OpenAIEmbeddings(model=model)
    if provider == "huggingface":
        if HuggingFaceEmbeddings is None:
            raise ImportError(
                "langchain-huggingface is not installed. Please install it to use HuggingFace embeddings."
            )
        # Pin to CPU by default. The embedding model is small (Qwen3-Embedding-0.6B
        # ≈ 1.2 GB) and is only called a handful of times per case for FAISS
        # retrieval, so GPU placement isn't worth the cost. More importantly,
        # vLLM workers spawn fresh Python processes and re-import this module —
        # if the embedding loaded on GPU 0, every worker would replicate it
        # there (~17 GB on rank 0), starving the vLLM engine of memory.
        # Override with FOAMAGENT_EMBED_DEVICE=cuda if you really want GPU.
        device = os.getenv("FOAMAGENT_EMBED_DEVICE", "cpu")
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": device},
        )
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model)

    raise ValueError(f"Unsupported embedding provider: {provider}")


def load_faiss_dbs(config: Optional[Config] = None):
    cfg = config or Config()
    embedding_model = get_embedding_model(cfg)

    base_dir = Path(__file__).resolve().parent.parent / "database" / "faiss"

    # Sanitize model name for directory usage
    model_dir_name = (cfg.embedding_model or "").replace("/", "_").replace(":", "_")
    db_path = base_dir / model_dir_name

    print(f"Loading FAISS indices from: {db_path} with model: {cfg.embedding_model}")

    dbs = {}
    indices = [
        "openfoam_allrun_scripts",
        "openfoam_tutorials_structure",
        "openfoam_tutorials_details",
        "openfoam_command_help",
    ]

    for index in indices:
        index_path = db_path / index
        if index_path.exists():
            try:
                dbs[index] = FAISS.load_local(
                    str(index_path), embedding_model, allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Failed to load index {index}: {e}")
        else:
            print(f"Warning: Index path does not exist: {index_path}")

    return dbs


# Lazy-loaded cache. Populated on first call to retrieve_faiss(), NOT at import time.
# Why: vLLM uses the `spawn` multiprocessing start method, so each worker process
# re-imports this module. If we eager-loaded the embedding model + FAISS indices
# at module import, every worker would replicate ~3-4 GB on GPU 0 (×4 workers ≈
# 17 GB), causing vLLM's startup memory check to fail with
# "Free memory on device cuda:0 (62.27/79.25 GiB) on startup is less than ...".
# Workers never call retrieve_faiss(), so this stays empty in worker processes.
FAISS_DB_CACHE = {}
_FAISS_LOAD_LOCK = threading.Lock()


def _ensure_faiss_loaded():
    """Populate FAISS_DB_CACHE on first use. Safe to call repeatedly."""
    global FAISS_DB_CACHE
    if FAISS_DB_CACHE:
        return
    with _FAISS_LOAD_LOCK:
        if FAISS_DB_CACHE:
            return
        FAISS_DB_CACHE = load_faiss_dbs()

class FoamfilePydantic(BaseModel):
    file_name: str = Field(description="Name of the OpenFOAM input file")
    folder_name: str = Field(description="Folder where the foamfile should be stored")
    content: str = Field(description="Content of the OpenFOAM file, written in OpenFOAM dictionary format")

class FoamPydantic(BaseModel):
    list_foamfile: List[FoamfilePydantic] = Field(description="List of OpenFOAM configuration files")

class ResponseWithThinkPydantic(BaseModel):
    think: str = Field(description="Thought process of the LLM")
    response: str = Field(description="Response of the LLM")
    
class _CodexResponsesWrapper:
    """Wrapper for an OpenAI Responses-compatible endpoint.

    This mimics the minimal interface LLMService expects from LangChain chat models:
    - invoke(messages) -> object with .content
    - get_num_tokens(text) -> int

    We support two wire endpoints:
    - OpenAI Platform: https://api.openai.com/v1/responses (API key / some OAuth tokens)
    - ChatGPT/Codex subscription backend: https://chatgpt.com/backend-api/codex/responses

    The ChatGPT backend requires a non-empty `instructions` field that matches the Codex harness
    expectations. We ship a default copy in `src/codex_instructions_default.txt`.
    """

    class _Resp:
        def __init__(self, content: str):
            self.content = content

    def __init__(
        self,
        token: str,
        model: str,
        temperature: float = 0.0,
        *,
        base_url: str = "https://api.openai.com/v1",
        account_id: Optional[str] = None,
        instructions: Optional[str] = None,
        stream: bool = False,
    ):
        self._token = token
        self._model = model
        self._temperature = temperature
        self._base_url = base_url.rstrip("/")
        self._account_id = account_id
        self._instructions = instructions
        self._stream = stream
        # Token counting (best-effort). Exact tokenization may differ by model.
        # We default to a modern tokenizer; adjust if you need model-specific counting.
        try:
            self._enc = tiktoken.get_encoding("o200k_base")
        except Exception:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def get_num_tokens(self, text: str) -> int:
        return len(self._enc.encode(text or ""))

    @staticmethod
    def _extract_json_object(text: str) -> str:
        """Best-effort extraction of a JSON object from a model response."""
        if not text:
            raise ValueError("Empty response; expected JSON")

        s = text.strip()
        # Strip fenced code blocks.
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
            s = re.sub(r"\n```\s*$", "", s).strip()

        # If it's already a JSON object, return it.
        if s.startswith("{") and s.endswith("}"):
            return s

        # Otherwise, find the first {...} block.
        m = re.search(r"\{[\s\S]*\}", s)
        if not m:
            raise ValueError(f"Could not find a JSON object in response: {s[:200]}")
        return m.group(0)

    def with_structured_output(self, pydantic_obj: Type[BaseModel]):
        """Return a wrapper that parses the model output into a Pydantic object.

        This is a minimal compatibility shim for LangChain's `.with_structured_output()`.
        """

        parent = self

        class _StructuredWrapper:
            def get_num_tokens(self, text: str) -> int:
                return parent.get_num_tokens(text)

            def invoke(self, messages):
                schema = pydantic_obj.model_json_schema()
                schema_hint = (
                    "Return ONLY valid JSON (no markdown) that matches this JSON Schema:\n"
                    + str(schema)
                )

                patched = list(messages)
                # Prepend a system constraint for JSON output.
                patched.insert(0, {"role": "system", "content": schema_hint})

                resp = parent.invoke(patched)
                raw = getattr(resp, "content", "")
                json_text = parent._extract_json_object(raw)
                return pydantic_obj.model_validate_json(json_text)

        return _StructuredWrapper()

    @staticmethod
    def _to_responses_input(messages):
        out = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            # Responses API supports rich content; we use simple input_text.
            out.append({"role": role, "content": [{"type": "input_text", "text": content}]})
        return out

    @staticmethod
    def _extract_output_text(resp_json: dict) -> str:
        # Newer APIs often include output_text; fall back to traversing.
        if isinstance(resp_json, dict) and isinstance(resp_json.get("output_text"), str):
            return resp_json["output_text"]

        texts = []
        for item in resp_json.get("output", []) if isinstance(resp_json, dict) else []:
            for c in item.get("content", []) if isinstance(item, dict) else []:
                if isinstance(c, dict):
                    if c.get("type") in {"output_text", "text"} and isinstance(c.get("text"), str):
                        texts.append(c["text"])
        return "\n".join(texts).strip()

    def _build_payload(self, messages):
        payload = {
            "model": self._model,
            "input": self._to_responses_input(messages),
        }

        # OpenAI Platform supports temperature.
        if "chatgpt.com" not in self._base_url:
            payload["temperature"] = self._temperature

        # ChatGPT/Codex subscription backend expects these extra keys.
        if "chatgpt.com" in self._base_url:
            payload.update(
                {
                    "instructions": self._instructions or "",
                    "tools": [],
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                    "reasoning": {"summary": "auto"},
                    "store": False,
                    "stream": bool(self._stream),
                    "include": ["reasoning.encrypted_content"],
                }
            )
        return payload

    @staticmethod
    def _iter_sse_text(resp: requests.Response):
        """Yield decoded SSE 'data:' payloads as strings."""
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            line = str(raw).strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            yield data

    def invoke(self, messages):
        url = f"{self._base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json" if not self._stream else "text/event-stream",
            "User-Agent": "Foam-Agent",
        }
        if self._account_id:
            headers["ChatGPT-Account-Id"] = self._account_id

        payload = self._build_payload(messages)

        r = requests.post(url, headers=headers, json=payload, timeout=60, stream=bool(self._stream))

        # If we get an error, surface the response body to aid debugging.
        if not r.ok:
            try:
                detail = r.text[:2000]
            except Exception:
                detail = ""
            raise requests.HTTPError(
                f"HTTP {r.status_code} for {url}. Body: {detail}", response=r
            )

        if not self._stream:
            data = r.json()
            return self._Resp(self._extract_output_text(data))

        # Streaming: accumulate text deltas.
        import json

        chunks: list[str] = []
        for s in self._iter_sse_text(r):
            try:
                j = json.loads(s)
            except Exception:
                continue

            # Codex backend streams OpenAI Responses-style events.
            if isinstance(j, dict):
                t = j.get("type")
                if t == "response.output_text.delta" and isinstance(j.get("delta"), str):
                    chunks.append(j["delta"])
                    continue
                if t == "response.output_text.done" and isinstance(j.get("text"), str):
                    # Some clients rely on done; we already collected deltas but keep as fallback.
                    if not chunks:
                        chunks.append(j["text"])
                    continue

            # Fallback: try generic extraction.
            t2 = self._extract_output_text(j)
            if t2:
                chunks.append(t2)

        return self._Resp("".join(chunks).strip())


# -------------------------
# Local HuggingFace (vLLM)
# -------------------------
# Minimal chat wrapper so LLMService can treat vLLM like a LangChain chat model.
# Supports: invoke(messages)->.content, get_num_tokens(text), with_structured_output(...)

_VLLM_MODEL_CACHE: Dict[str, Any] = {}
_VLLM_TOKENIZER_CACHE: Dict[str, Any] = {}
_VLLM_LOCKS: Dict[str, threading.Lock] = {}
_VLLM_INIT_LOCK = threading.Lock()
_VLLM_LORA_ENABLED: set = set()  # track which cached models were loaded with enable_lora=True

# Architecture-keyed engine cache for the in-place weight-swap path.
# Two checkpoint paths share an engine iff their config.json fields produce
# the same arch_key (architecture, hidden_size, vocab_size, num_hidden_layers,
# dtype). For the 9 gpt-oss-20b merged checkpoints this collapses 9
# engines into 1; weights are reloaded in place on substep transitions.
_VLLM_ARCH_ENGINE: Dict[str, Any] = {}        # arch_key -> LLM
_VLLM_CURRENT_PATH: Dict[int, str] = {}        # id(LLM) -> currently-loaded weights path
_VLLM_ARCH_LOCKS: Dict[str, threading.Lock] = {}


def _arch_key_for(model_path: str) -> str:
    """Stable key over architecture-shaping fields only.

    Two paths produce the same key iff their checkpoints share the same
    Worker buffer layout (architecture, hidden size, vocab size, dtype).
    Falls back to the path itself if the config can't be read — in that
    case the wrapper degrades to per-path engines (the legacy behavior).
    """
    cfg_path = os.path.join(model_path, "config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception:
        return model_path
    archs = cfg.get("architectures") or ["?"]
    return "|".join([
        archs[0] if isinstance(archs, list) and archs else "?",
        str(cfg.get("hidden_size")),
        str(cfg.get("vocab_size")),
        str(cfg.get("num_hidden_layers")),
        str(cfg.get("torch_dtype") or cfg.get("dtype")),
    ])


def _get_or_create_arch_lock(arch_key: str) -> threading.Lock:
    lock = _VLLM_ARCH_LOCKS.get(arch_key)
    if lock is None:
        lock = threading.Lock()
        _VLLM_ARCH_LOCKS[arch_key] = lock
    return lock


def _build_vllm_engine(
    model_path: str,
    *,
    tensor_parallel_size: int = 1,
    dtype: str = "bfloat16",
    quantization: Optional[str] = None,
    max_model_len: Optional[int] = None,
    need_lora: bool = False,
):
    """Build a vLLM engine from env-driven kwargs.

    Single source of truth for both engine cache modes:
      - the legacy per-path cache (LoRA / Nemotron) and
      - the arch-keyed in-place swap cache (gpt-oss merged checkpoints).

    For the swap path (need_lora=False) we default to enforce_eager=True:
    CUDA graph capture is tied to specific tensor-storage layouts that
    reload_weights() leaves untouched in principle, but skipping graphs
    eliminates a class of subtle replay issues and frees ~2 GB. The LoRA
    path keeps the historical enforce_eager=False default.
    """
    from vllm import LLM

    gpu_mem_util = float(os.getenv("FOAMAGENT_HF_GPU_MEM_UTIL", "0.90"))
    max_num_seqs = int(os.getenv("FOAMAGENT_HF_MAX_NUM_SEQS", "1"))
    enforce_eager_default = "1" if not need_lora else "0"
    enforce_eager = os.getenv("FOAMAGENT_HF_ENFORCE_EAGER", enforce_eager_default).lower() in ("1", "true", "yes")
    swap_space_gb = int(os.getenv("FOAMAGENT_HF_SWAP_SPACE_GB", "4"))
    kv_cache_dtype = os.getenv("FOAMAGENT_HF_KV_CACHE_DTYPE")
    llm_kwargs = dict(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        tensor_parallel_size=int(tensor_parallel_size),
        dtype=dtype,
        quantization=quantization,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        max_num_seqs=max_num_seqs,
        enforce_eager=enforce_eager,
        swap_space=swap_space_gb,
    )
    if kv_cache_dtype:
        llm_kwargs["kv_cache_dtype"] = kv_cache_dtype
    # kit_e49 diagnostic (2026-10-10): with FOAMAGENT_HF_LOG_STATS=1 the engine logs its throughput every few seconds
    # (vLLM's stats logger; outputs unchanged), so a call stopped by the 3 h wall guard shows whether it was still
    # generating (runaway thinking) or had stalled
    if os.getenv("FOAMAGENT_HF_LOG_STATS") == "1":
        llm_kwargs["disable_log_stats"] = False
    # FoamGPT E46-E48 (kit_e46): swap_space is not an engine argument in vLLM 0.20.1 either; there LLM() popped it
    # with "deprecated and ignored", so E32-E45 never used it. vLLM 0.31 removed that shim and EngineArgs rejects the
    # keyword, so it is dropped here whenever EngineArgs does not take it: the engine gets the same arguments as before.
    import inspect as _inspect
    from vllm.engine.arg_utils import EngineArgs as _EngineArgs
    if "swap_space" not in _inspect.signature(_EngineArgs).parameters:
        llm_kwargs.pop("swap_space")
    if need_lora:
        max_lora_rank = int(os.getenv("FOAMAGENT_HF_MAX_LORA_RANK", "16"))
        max_loras = int(os.getenv("FOAMAGENT_HF_MAX_LORAS", "9"))
        llm_kwargs.update(
            enable_lora=True,
            max_loras=max_loras,
            max_lora_rank=max_lora_rank,
        )
        print(f"  Loading vLLM engine with LoRA support (max_loras={max_loras}, max_lora_rank={max_lora_rank})")
    print(
        f"  vLLM engine: tp={tensor_parallel_size} max_model_len={max_model_len} "
        f"max_num_seqs={max_num_seqs} gpu_mem_util={gpu_mem_util} "
        f"enforce_eager={enforce_eager} swap_space_gb={swap_space_gb} "
        f"kv_cache_dtype={kv_cache_dtype or 'auto'}"
    )
    return LLM(**llm_kwargs)


# ---------------------------------------------------------------------------------------------------------------------
# FoamGPT kit_e49 (E49-E52, thinking on). Inactive unless the run's
# environment asks for it:
#   FOAMAGENT_HF_OUTPUT_CAP=context  each call's output cap = max_model_len - prompt tokens - 64 (instead of the fixed
#                                    FOAMAGENT_HF_MAX_OUTPUT_TOKENS, which still sets the prompt-fitting reserve, so
#                                    prompts are fitted exactly as in the thinking-off runs)
#   FOAMAGENT_HF_CALL_WALL_S=10800   watchdog around each generation: a call still running after this many seconds ends
#                                    the case as the model's own ending ("[wall-guard]"); the E47/E48 engine-deadlock
#                                    signature (one GPU < 20 % busy while the other three are >= 90 %) in
#                                    FOAMAGENT_HF_HANG_MIN (30) consecutive one-minute samples ends it as an
#                                    infrastructure failure ("[engine-hang]", set aside by auto-triage for a re-run alone)
# Every generation is logged to calls.jsonl next to the case's dataset.jsonl (call site, prompt/output tokens, cap,
# finish reason, seconds, thinking tokens, whether the thinking block was closed). When the rendered prompt itself opens
# the thinking block (thinking-mode templates end with "<think>"), the output starts inside it, so "<think>" is put
# back in front before the harness's own stripping removes the block; an unclosed block is left as the model wrote it.
_CALL_CTX = threading.local()


def _gpu_utils():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60).stdout
        return [int(x) for x in out.split()]
    except Exception:
        return []


def _kill_children_and_exit(rc, msg):
    import sys as _sys
    import psutil
    print(msg, flush=True)
    _sys.stdout.flush()
    _sys.stderr.flush()
    kids = psutil.Process().children(recursive=True)
    for k in kids:
        try:
            k.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(kids, timeout=30)
    os._exit(rc)


class _CallWatchdog:
    def __init__(self, site):
        self.site = site
        self.wall = float(os.getenv("FOAMAGENT_HF_CALL_WALL_S", "0") or 0)
        # engine-hang detector OFF by default (2026-10-08): the "one GPU idle, three spinning" pattern is how this
        # eager-mode TP4 engine looks during normal slow decoding (DeepSeek V4 at ~4.5 tokens/s), not only in a hang
        self.hang_min = int(os.getenv("FOAMAGENT_HF_HANG_MIN", "0"))
        self.t0 = time.time()
        self.hang_streak = 0
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        if self.wall > 0:
            self._th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(float(os.getenv("FOAMAGENT_HF_WATCH_S", "60"))):
            el = time.time() - self.t0
            u = _gpu_utils()
            s = sorted(u)
            hang = len(s) >= 4 and s[0] < 20 and all(x >= 90 for x in s[1:])
            self.hang_streak = self.hang_streak + 1 if hang else 0
            if self.hang_min > 0 and self.hang_streak >= self.hang_min:
                _kill_children_and_exit(3, f"[engine-hang] call site {self.site}: GPU utilization {u}, one GPU idle while "
                                           f"the others spin, for {self.hang_streak} consecutive minutes; call killed after "
                                           f"{el / 60:.0f} min (infrastructure, not the model's ending)")
            if el >= self.wall:
                _kill_children_and_exit(4, f"[wall-guard] call site {self.site}: one generation exceeded "
                                           f"{self.wall / 3600:.1f} h (GPU utilization {u}); counted as the model's own "
                                           f"ending\nWorkflow failed with error: wall-guard")


class _VLLMChatWrapper:
    class _Resp:
        def __init__(self, content: str):
            self.content = content

    _lora_id_counter = 0  # class-level counter for unique LoRA int IDs

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        *,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        quantization: Optional[str] = None,
        max_model_len: Optional[int] = None,
        lora_adapter_path: Optional[str] = None,
        chat_template_path: Optional[str] = None,
    ):
        self._model = model
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._max_model_len = int(max_model_len) if max_model_len else None
        self._chat_template_path = chat_template_path

        # Lazy imports to avoid hard dependency unless used.
        try:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except Exception as e:
            raise ImportError(
                "model_provider='huggingface' requires transformers + vllm installed"
            ) from e

        self._SamplingParams = SamplingParams

        # LoRA adapter support: create a LoRARequest if an adapter path is given.
        # vLLM loads the base model once and swaps LoRA weights per-request.
        self._lora_request = None
        if lora_adapter_path:
            from vllm.lora.request import LoRARequest
            _VLLMChatWrapper._lora_id_counter += 1
            # Use the parent directory name as the adapter name (e.g. "parse_case_info")
            adapter_name = os.path.basename(os.path.dirname(lora_adapter_path)) or os.path.basename(lora_adapter_path)
            self._lora_request = LoRARequest(
                lora_name=adapter_name,
                lora_int_id=_VLLMChatWrapper._lora_id_counter,
                lora_path=str(os.path.abspath(lora_adapter_path)),
            )
            print(f"  LoRA adapter registered: {adapter_name} (id={_VLLMChatWrapper._lora_id_counter}) from {lora_adapter_path}")

        force_lora_engine = os.getenv("FOAMAGENT_HF_FORCE_LORA_ENGINE", "0").lower() in (
            "1", "true", "yes"
        )
        need_lora = lora_adapter_path is not None or force_lora_engine

        # Two engine-cache modes:
        #   swap mode (default for plain merged checkpoints): one engine per
        #     architecture, weights swapped in place via reload_weights() as
        #     wrappers for different paths take turns running. Collapses the 9
        #     gpt-oss-20b substep checkpoints into a single resident engine.
        #   legacy mode (LoRA path or swap explicitly disabled): one engine per
        #     model path. Required for the Nemotron LoRA adapters because vLLM
        #     mounts adapters onto a single base engine via LoRARequest.
        # FOAMAGENT_HF_FORCE_LORA_ENGINE keeps LoRA support enabled even for a
        # bare-base request. This lets an ablation share one engine between
        # adapter-backed call sites and the single call site routed to base.
        swap_enabled = os.getenv("FOAMAGENT_HF_INPLACE_SWAP", "1").lower() in ("1", "true", "yes")
        # FoamGPT E48 container fallback: with FOAMAGENT_HF_REMOTE_URL set, no engine is
        # built here; the engine is vLLM's OpenAI-compatible server (official image, Slurm container on the same node)
        # and each call sends it the prompt rendered below with the same sampling settings (_remote_generate).
        self._remote_url = os.getenv("FOAMAGENT_HF_REMOTE_URL")
        if self._remote_url and need_lora:
            raise ValueError("FOAMAGENT_HF_REMOTE_URL: LoRA adapters are not supported over HTTP")
        use_swap = swap_enabled and not need_lora and not self._remote_url
        self._use_swap = use_swap
        self._target_path = model

        # Ensure a single tokenizer/model instantiation per process (thread-safe).
        with _VLLM_INIT_LOCK:
            tok = _VLLM_TOKENIZER_CACHE.get(model)
            if tok is None:
                tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
                _VLLM_TOKENIZER_CACHE[model] = tok
            # Override the chat template with the one used during training. Required
            # for Gemma 4 fine-tunes: the native template injects
            # `<|channel>thought\n<channel|>` after `<|turn>model\n` whenever
            # enable_thinking=False, but training used the simpler unsloth
            # `gemma-4` alias that ends the prompt at `<|turn>model\n`. Without
            # this override the LoRA sees a prompt suffix it was never trained on.
            if chat_template_path:
                with open(chat_template_path, "r", encoding="utf-8") as f:
                    tok.chat_template = f.read()
                print(f"  Overrode chat template from {chat_template_path}")
            self._tok = tok

            if self._remote_url:
                self._llm = None
                lock = _VLLM_LOCKS.get(model)
                if lock is None:
                    lock = threading.Lock()
                    _VLLM_LOCKS[model] = lock
                self._lock = lock
                print(f"  Engine served over HTTP: {self._remote_url} (no in-process vLLM engine)")
            elif use_swap:
                arch_key = _arch_key_for(model)
                self._arch_key = arch_key
                self._lock = _get_or_create_arch_lock(arch_key)

                llm = _VLLM_ARCH_ENGINE.get(arch_key)
                if llm is None:
                    print(f"  Building shared vLLM engine for arch={arch_key} (init weights: {os.path.basename(model)})")
                    llm = _build_vllm_engine(
                        model_path=model,
                        tensor_parallel_size=tensor_parallel_size,
                        dtype=dtype,
                        quantization=quantization,
                        max_model_len=max_model_len,
                        need_lora=False,
                    )
                    _VLLM_ARCH_ENGINE[arch_key] = llm
                    _VLLM_CURRENT_PATH[id(llm)] = model
                self._llm = llm
            else:
                llm = _VLLM_MODEL_CACHE.get(model)

                # If we need LoRA but the cached model was loaded without it, recreate.
                if llm is not None and need_lora and model not in _VLLM_LORA_ENABLED:
                    print(f"  ↪ Recreating vLLM engine for {model} with enable_lora=True")
                    del _VLLM_MODEL_CACHE[model]
                    llm = None

                if llm is None:
                    llm = _build_vllm_engine(
                        model_path=model,
                        tensor_parallel_size=tensor_parallel_size,
                        dtype=dtype,
                        quantization=quantization,
                        max_model_len=max_model_len,
                        need_lora=need_lora,
                    )
                    _VLLM_MODEL_CACHE[model] = llm
                    if need_lora:
                        _VLLM_LORA_ENABLED.add(model)
                self._llm = llm

                lock = _VLLM_LOCKS.get(model)
                if lock is None:
                    lock = threading.Lock()
                    _VLLM_LOCKS[model] = lock
                self._lock = lock

        # Per-thread accumulator for the vLLM engine's own accounting of the
        # last outer invoke (set by the direct path; summed across retries by
        # with_structured_output). LLMService reads + clears this via
        # pop_last_usage() so cache-hit stats end up in output.txt.
        self._tl = threading.local()

    def _reset_usage(self) -> None:
        self._tl.cached_prompt_tokens = 0
        self._tl.engine_prompt_tokens = 0
        self._tl.engine_completion_tokens = 0

    def _add_usage(self, cached: int, prompt: int, completion: int) -> None:
        if not hasattr(self._tl, "cached_prompt_tokens"):
            self._reset_usage()
        self._tl.cached_prompt_tokens += int(cached or 0)
        self._tl.engine_prompt_tokens += int(prompt or 0)
        self._tl.engine_completion_tokens += int(completion or 0)

    def pop_last_usage(self) -> dict:
        usage = {
            "cached_prompt_tokens": getattr(self._tl, "cached_prompt_tokens", 0),
            "engine_prompt_tokens": getattr(self._tl, "engine_prompt_tokens", 0),
            "engine_completion_tokens": getattr(self._tl, "engine_completion_tokens", 0),
        }
        self._reset_usage()
        return usage

    def _ensure_active(self) -> None:
        """Swap engine weights to self._target_path if a different path is loaded.

        No-op when not in swap mode (LoRA path / swap disabled). Caller must
        hold self._lock so that no concurrent generation runs against stale
        weights. The prefix-cache reset is mandatory: blocks were computed
        under the old weights and would otherwise yield silent wrong outputs.
        """
        if not self._use_swap:
            return
        cur = _VLLM_CURRENT_PATH.get(id(self._llm))
        if cur == self._target_path:
            return
        t0 = time.perf_counter()
        self._llm.collective_rpc(
            "reload_weights",
            kwargs={
                "weights_path": self._target_path,
                "is_checkpoint_format": True,
            },
        )
        self._llm.reset_prefix_cache()
        _VLLM_CURRENT_PATH[id(self._llm)] = self._target_path
        elapsed = time.perf_counter() - t0
        print(f"  [swap] weights -> {os.path.basename(self._target_path)} in {elapsed:.1f}s")

    def get_num_tokens(self, text: str) -> int:
        return len(self._tok.encode(text or ""))

    def _messages_to_prompt(self, messages):
        """Format messages using the tokenizer's chat template.

        This ensures the correct special tokens are used for each model
        (e.g. Harmony tokens for GPT-OSS, ChatML for Nemotron/Qwen, etc.).
        """
        # GPT-OSS Harmony accepts reasoning_effort; Nemotron-3-Nano accepts
        # enable_thinking (defaults True in its template — leaves reasoning prose
        # in the final channel and breaks JSON parsing). Try richest kwargs
        # first, drop whichever the active template rejects.
        reasoning_effort = os.getenv("FOAMAGENT_HF_REASONING_EFFORT", "low")
        enable_thinking = os.getenv("FOAMAGENT_HF_ENABLE_THINKING", "false").lower() in {"1", "true", "yes"}
        for kwargs in (
            {"reasoning_effort": reasoning_effort, "enable_thinking": enable_thinking},
            {"enable_thinking": enable_thinking},
            {"reasoning_effort": reasoning_effort},
            {},
        ):
            try:
                return self._tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **kwargs
                )
            except TypeError:
                continue
            except Exception:
                break
        try:
            return self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback: plain text format if the tokenizer has no chat template.
            parts = []
            for m in messages:
                parts.append(f"{m.get('role', 'user')}: {m.get('content', '')}")
            parts.append("assistant:")
            return "\n".join(parts)

    @staticmethod
    def _strip_reasoning_channels(text: str) -> str:
        """Remove reasoning-channel content so downstream parsers see only the answer.

        Handles two patterns:
          1. <think>...</think>, <thinking>...</thinking>, <reasoning>...</reasoning>
             tags emitted by DeepSeek, QwQ, Nemotron, etc.
          2. GPT-OSS Harmony channels. After detokenization with skip_special_tokens=True,
             channel markers leak as bare text:
               "<|channel|>analysis<|message|>...<|channel|>final<|message|>..."
               -> "analysis...assistantfinal..." (or similar)
             We keep only the substring after the LAST 'final' marker.
        """
        if not text:
            return text
        # 1. Tag-based reasoning blocks.
        cleaned = re.sub(
            r"<(?:think|thinking|reasoning)>.*?</(?:think|thinking|reasoning)>\s*",
            "", text, flags=re.DOTALL,
        ).strip()
        # 2. GPT-OSS Harmony: take what follows the last 'final' channel marker.
        #    The model emits "...assistantfinal{response}" with NO separator
        #    (special tokens like <|channel|> get stripped during detokenization).
        #    We cannot constrain the prefix (it could be 'assistant', another
        #    channel name, or punctuation), so we just match 'final' wherever
        #    it appears as long as it's not part of 'finally' / 'finalize'.
        #    Negative lookahead (?![a-z]) guards against those words while
        #    still matching 'finalBelow', 'final{', 'final ', etc.
        #    Off by default: in the answers of models without Harmony channels the rule can cut valid JSON
        #    at a lowercase "final". The configs that reproduce runs made with it (the shared and split
        #    adapters, the untuned Nemotron-Super run E33) set FOAMGPT_STRIP_FINAL=1.
        if os.environ.get("FOAMGPT_STRIP_FINAL") == "1":
            markers = list(re.finditer(r"final(?![a-z])", cleaned))
            if markers:
                cleaned = cleaned[markers[-1].end():].lstrip(" \t\r\n:>")
        return cleaned

    @staticmethod
    def _extract_json_object(text: str) -> str:
        """Extract the first balanced JSON object from text.

        Uses brace counting that respects string literals. This handles:
          - models that emit prose before/after the JSON
          - models that emit two adjacent objects (we take the first complete one)
          - braces inside string content (e.g. OpenFOAM dict syntax in 'content' fields)
        """
        if not text:
            raise ValueError("Empty response; expected JSON")

        s = text.strip()
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", s)
            s = re.sub(r"\n?```\s*$", "", s).strip()

        start = s.find("{")
        if start < 0:
            raise ValueError(f"No JSON object found in response: {s[:200]}")

        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(s)):
            c = s[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]

        # Unbalanced — likely truncated. Return best-effort so callers can
        # report a more useful error than "regex didn't match".
        raise ValueError(
            f"JSON object is unbalanced (likely truncated). Depth={depth} after {len(s) - start} chars."
        )

    def with_structured_output(self, pydantic_obj: Type[BaseModel]):
        parent = self

        class _StructuredWrapper:
            def get_num_tokens(self, text: str) -> int:
                return parent.get_num_tokens(text)

            def invoke(self, messages):
                schema = pydantic_obj.model_json_schema()
                schema_hint = (
                    "Return ONLY valid JSON (no markdown) that matches this JSON Schema:\n"
                    + str(schema)
                )

                patched = list(messages)
                patched.insert(0, {"role": "system", "content": schema_hint})

                last_raw = ""
                for attempt in range(3):
                    resp = parent.invoke(patched)
                    raw = getattr(resp, "content", "")
                    # invoke() already strips reasoning channels, but apply again
                    # in case messages were passed through a different code path.
                    raw = parent._strip_reasoning_channels(raw)
                    last_raw = raw
                    try:
                        json_text = parent._extract_json_object(raw)
                        return pydantic_obj.model_validate_json(json_text)
                    except Exception as e:
                        # Ask the model to repair its JSON.
                        repair = (
                            "Your previous output was not valid JSON for the requested schema. "
                            "Return ONLY valid JSON (no markdown), with no extra keys. "
                            f"Error: {str(e)}\n\n"
                            f"Previous output:\n{raw}"
                        )
                        patched = list(patched) + [{"role": "user", "content": repair}]

                raise ValueError(
                    f"Failed to produce valid JSON after retries. Last output: {last_raw[:500]}"
                )

        return _StructuredWrapper()

    def invoke(self, messages):
        messages = self._fit_to_context(list(messages))
        prompt = self._messages_to_prompt(messages)

        # Final guard: if even after content trimming the rendered prompt is
        # still too long (chat-template overhead, system prompt, etc.), shrink
        # the requested output budget so the request doesn't violate
        # max_model_len. We never let the request hit vLLM's hard limit.
        max_tokens = self._max_tokens
        if os.getenv("FOAMAGENT_HF_OUTPUT_CAP") == "context" and self._max_model_len is not None:
            max_tokens = self._max_model_len  # kit_e49: lowered below to max_model_len - prompt - 64
        if self._max_model_len is not None:
            prompt_token_count = len(self._tok.encode(prompt))
            available = self._max_model_len - prompt_token_count - 64  # safety
            if available < max_tokens:
                if available < 256:
                    raise ValueError(
                        f"Prompt is {prompt_token_count} tokens, leaving only {available} for "
                        f"output (max_model_len={self._max_model_len}). Cannot generate."
                    )
                max_tokens = available

        if self._remote_url:
            with self._lock:
                text, cached, eng_prompt, eng_completion = self._remote_generate(prompt, max_tokens)
            self._add_usage(cached, eng_prompt, eng_completion)
            text = self._strip_reasoning_channels(text or "")
            return self._Resp(text.strip())

        params = self._SamplingParams(
            temperature=self._temperature,
            max_tokens=max_tokens,
        )

        # vLLM engine is not safe to call concurrently from multiple threads.
        # Serialize generation per model (legacy mode) or per architecture
        # (swap mode). The lock also guards _ensure_active() so that a swap
        # never races with another wrapper's generate() on the same engine —
        # concurrent generation against in-flight reload_weights() would
        # silently produce wrong outputs.
        site = getattr(_CALL_CTX, "substep", "")
        t_gen = time.time()
        with self._lock, _CallWatchdog(site):
            self._ensure_active()
            gen_kwargs = {}
            if self._lora_request is not None:
                gen_kwargs["lora_request"] = self._lora_request
            out = self._llm.generate([prompt], params, **gen_kwargs)[0]
        t_gen = time.time() - t_gen

        # Record the engine's own token accounting for this call so LLMService
        # can report prefix-cache hit rate. num_cached_tokens is the count of
        # prompt tokens served from vLLM's prefix cache (equivalent to the
        # OpenAI usage field prompt_tokens_details.cached_tokens).
        cached = int(getattr(out, "num_cached_tokens", 0) or 0)
        eng_prompt = len(getattr(out, "prompt_token_ids", []) or [])
        eng_completion = sum(
            len(getattr(o, "token_ids", []) or [])
            for o in (getattr(out, "outputs", []) or [])
        )
        self._add_usage(cached, eng_prompt, eng_completion)

        text = out.outputs[0].text if getattr(out, "outputs", None) else ""
        raw = text or ""
        # kit_e49: the thinking-mode prompt opens the block itself; put the tag back so the stripping below removes it
        think_open = prompt.rstrip().endswith("<think>")
        closed = "</think>" in raw
        if think_open and closed:
            text = "<think>" + raw
        if think_open:
            thinking_tokens = (len(self._tok.encode(raw.split("</think>", 1)[0], add_special_tokens=False)) if closed
                               else eng_completion)
        else:
            thinking_tokens = 0
        log_dir = getattr(_CALL_CTX, "log_dir", None)
        if log_dir:
            rec = {"site": site, "prompt_tokens": eng_prompt, "max_tokens": max_tokens,
                   "completion_tokens": eng_completion,
                   "finish_reason": getattr(out.outputs[0], "finish_reason", None) if getattr(out, "outputs", None) else None,
                   "seconds": round(t_gen, 1), "think_open": think_open, "think_closed": closed,
                   "thinking_tokens": thinking_tokens, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            try:
                with open(os.path.join(log_dir, "calls.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except OSError:
                pass
        text = self._strip_reasoning_channels(text or "")
        return self._Resp(text.strip())

    def _remote_generate(self, prompt: str, max_tokens: int):
        """E48 container fallback: one completion from the local vLLM server for the prompt rendered by
        _messages_to_prompt, with the in-process path's sampling settings (temperature, max_tokens; everything else at
        vLLM's defaults: the server runs with --generation-config vllm). No retries: a failed request ends the call."""
        import json as _json
        import urllib.request as _ureq
        body = {"model": os.getenv("FOAMAGENT_HF_REMOTE_MODEL", "engine"), "prompt": prompt,
                "temperature": self._temperature, "max_tokens": int(max_tokens), "n": 1, "stream": False}
        req = _ureq.Request(self._remote_url.rstrip("/") + "/v1/completions", data=_json.dumps(body).encode(),
                            headers={"Content-Type": "application/json"})
        with _ureq.urlopen(req, timeout=float(os.getenv("FOAMAGENT_HF_REMOTE_TIMEOUT", "7200"))) as r:
            d = _json.loads(r.read())
        u = d.get("usage") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        return d["choices"][0]["text"], int(cached), int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)

    def _fit_to_context(self, messages):
        """If the rendered prompt would exceed max_model_len, truncate the
        longest user/assistant message from the middle so head + tail (which
        usually contain the task framing) stay intact.

        OpenFOAM rewrite loops accumulate huge file dumps + error logs in the
        user message; truncation prevents the workflow from dying on
        'maximum context length exceeded' after a few rewrite iterations.
        """
        if self._max_model_len is None or not messages:
            return messages

        # Reserve room for the response.
        budget = self._max_model_len - self._max_tokens - 256  # safety for chat-template
        if budget <= 0:
            return messages

        # Render once to measure.
        try:
            current = len(self._tok.encode(self._messages_to_prompt(messages)))
        except Exception:
            return messages
        if current <= budget:
            return messages

        # Need to shed `current - budget` tokens. Pick the longest non-system
        # message and truncate its middle.
        non_system_idx = [i for i, m in enumerate(messages) if m.get("role") != "system"]
        if not non_system_idx:
            return messages
        target = max(non_system_idx, key=lambda i: len(self._tok.encode(messages[i].get("content", "") or "")))

        content = messages[target].get("content", "") or ""
        tokens = self._tok.encode(content)
        # Keep room for a clear marker.
        marker = "\n\n[... TRUNCATED to fit context window ...]\n\n"
        marker_tokens = len(self._tok.encode(marker))
        # New target token count for this message.
        excess = current - budget
        new_len = max(256, len(tokens) - excess - marker_tokens)
        if new_len >= len(tokens):
            return messages
        head_len = new_len // 2
        tail_len = new_len - head_len
        new_tokens = tokens[:head_len] + self._tok.encode(marker) + tokens[-tail_len:]
        new_content = self._tok.decode(new_tokens, skip_special_tokens=True)
        new_messages = list(messages)
        new_messages[target] = dict(messages[target])
        new_messages[target]["content"] = new_content
        print(
            f"  [context-fit] truncated message {target} from "
            f"{len(tokens)} -> {len(self._tok.encode(new_content))} tokens "
            f"(budget {budget}, was {current})"
        )
        return new_messages


# -------------------------
# Gemma 4 26B A4B (specialized)
# -------------------------
# Gemma 4 26B A4B's HF tokenizer_config.json sets
#   "extra_special_tokens": ["<|video|>"]
# but transformers <= 4.57 expects a dict and crashes inside
# `_set_model_specific_special_tokens` with
#   AttributeError: 'list' object has no attribute 'keys'
# Both AutoTokenizer and vLLM's internal tokenizer init hit this. The wrapper
# below side-steps the broken config entirely:
#   - Loads tokenizer.json directly via PreTrainedTokenizerFast (no Gemma init).
#   - Renders the unsloth `gemma-4` chat template (saved per LoRA adapter dir
#     as chat_template.jinja) with jinja2 — same prompt format training used.
#   - Hands vLLM a patched tokenizer dir so its internal AutoTokenizer call
#     finds extra_special_tokens as a dict.
def _patched_gemma4_tokenizer_dir(source_dir: str) -> str:
    """Materialize a transformers-friendly tokenizer dir from a saved adapter dir.

    Copies tokenizer.json, special_tokens_map.json, and chat_template.jinja
    verbatim, then writes a sanitized tokenizer_config.json where
    `extra_special_tokens` is coerced to a dict (transformers <= 4.57 rejects
    the list shape Gemma 4 saves). Idempotent: returns the same dir on repeat
    calls and only rewrites if the source mtime is newer.
    """
    import hashlib
    import shutil
    import tempfile

    src = os.path.abspath(source_dir)
    digest = hashlib.md5(src.encode("utf-8")).hexdigest()[:16]
    dest = os.path.join(tempfile.gettempdir(), "foamgpt_gemma4_tok", digest)
    os.makedirs(dest, exist_ok=True)

    # Required: tokenizer.json. Without it we can't load a fast tokenizer.
    src_tok = os.path.join(src, "tokenizer.json")
    if not os.path.isfile(src_tok):
        raise FileNotFoundError(f"tokenizer.json not found in {src}")
    dst_tok = os.path.join(dest, "tokenizer.json")
    if not os.path.isfile(dst_tok) or os.path.getmtime(src_tok) > os.path.getmtime(dst_tok):
        shutil.copy2(src_tok, dst_tok)

    # Patch tokenizer_config.json.
    src_cfg = os.path.join(src, "tokenizer_config.json")
    if os.path.isfile(src_cfg):
        with open(src_cfg, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if isinstance(cfg.get("extra_special_tokens"), list):
            cfg["extra_special_tokens"] = {}
        # processor_class triggers Gemma4Processor multimodal init that we
        # don't need for text-only inference; drop it.
        cfg.pop("processor_class", None)
        with open(os.path.join(dest, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    # Optional companion files.
    for fname in ("special_tokens_map.json", "chat_template.jinja"):
        src_f = os.path.join(src, fname)
        dst_f = os.path.join(dest, fname)
        if os.path.isfile(src_f) and (
            not os.path.isfile(dst_f) or os.path.getmtime(src_f) > os.path.getmtime(dst_f)
        ):
            shutil.copy2(src_f, dst_f)

    return dest


class _VLLMGemma4ChatWrapper:
    """vLLM wrapper specialized for Gemma 4 26B A4B.

    Differs from `_VLLMChatWrapper` in two ways:
      1. Tokenizer is loaded as PreTrainedTokenizerFast(tokenizer_file=...) —
         this skips the GemmaTokenizer init that crashes on the list-shaped
         `extra_special_tokens` field.
      2. The chat template is rendered with jinja2 directly from the LoRA
         adapter's chat_template.jinja (the unsloth `gemma-4` alias used at
         training time), not via `tokenizer.apply_chat_template`.

    `lora_adapter_path` is optional: when present, vLLM loads the base model
    with LoRA enabled and applies that adapter per request; when absent, the
    same wrapper runs the base model directly for baseline comparisons.
    """

    class _Resp:
        def __init__(self, content: str):
            self.content = content

    _lora_id_counter = 0

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        *,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        quantization: Optional[str] = None,
        max_model_len: Optional[int] = None,
        lora_adapter_path: Optional[str] = None,
        chat_template_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ):
        self._model = model
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._max_model_len = int(max_model_len) if max_model_len else None

        try:
            from transformers import PreTrainedTokenizerFast
            from vllm import SamplingParams
            import jinja2
        except Exception as e:
            raise ImportError(
                "model_type='gemma4' requires transformers + vllm + jinja2 installed"
            ) from e
        self._SamplingParams = SamplingParams

        # Fine-tuned runs use the adapter dir as the canonical source for both
        # tokenizer and chat template. Base runs use the model snapshot itself.
        template_source_dir = tokenizer_path or lora_adapter_path or model
        if not chat_template_path:
            chat_template_path = os.path.join(template_source_dir, "chat_template.jinja")
        if not os.path.isfile(chat_template_path):
            raise FileNotFoundError(f"chat_template.jinja not found at {chat_template_path}")

        patched_dir = _patched_gemma4_tokenizer_dir(template_source_dir)
        print(f"  [gemma4] patched tokenizer dir: {patched_dir}")

        # Our own tokenizer for length checks and token counting. Uses only
        # tokenizer.json and skips the broken Gemma init path.
        self._tok = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(patched_dir, "tokenizer.json")
        )
        # Gemma 4 uses these standard tokens; PreTrainedTokenizerFast leaves
        # them unset because we bypassed the model-specific init.
        self._bos_token = "<bos>"
        self._eos_token = "<eos>"

        # Pre-compile the chat template once.
        with open(chat_template_path, "r", encoding="utf-8") as f:
            template_src = f.read()

        def _raise(msg):
            raise jinja2.TemplateError(msg)

        env = jinja2.Environment(keep_trailing_newline=False)
        env.globals["raise_exception"] = _raise
        self._chat_template = env.from_string(template_src)

        # Optional LoRA registration (same shape as the generic wrapper).
        self._lora_request = None
        need_lora = bool(lora_adapter_path)
        if need_lora:
            from vllm.lora.request import LoRARequest
            _VLLMGemma4ChatWrapper._lora_id_counter += 1
            adapter_name = (
                os.path.basename(os.path.dirname(lora_adapter_path))
                or os.path.basename(lora_adapter_path)
            )
            self._lora_request = LoRARequest(
                lora_name=adapter_name,
                lora_int_id=_VLLMGemma4ChatWrapper._lora_id_counter,
                lora_path=str(os.path.abspath(lora_adapter_path)),
            )
            print(
                f"  [gemma4] LoRA adapter registered: {adapter_name} "
                f"(id={_VLLMGemma4ChatWrapper._lora_id_counter}) from {lora_adapter_path}"
            )

        # Build the vLLM engine. Cache is keyed on (base_model, patched_dir,
        # LoRA-enabled?) because all 9 substep wrappers for a base benchmark
        # share one resident copy, while fine-tuned runs need an engine with
        # enable_lora=True.
        cache_key = f"gemma4::{model}::{patched_dir}::lora={int(need_lora)}"
        with _VLLM_INIT_LOCK:
            llm = _VLLM_MODEL_CACHE.get(cache_key)
            if llm is None:
                from vllm import LLM
                gpu_mem_util = float(os.getenv("FOAMAGENT_HF_GPU_MEM_UTIL", "0.90"))
                max_num_seqs = int(os.getenv("FOAMAGENT_HF_MAX_NUM_SEQS", "1"))
                enforce_eager = os.getenv("FOAMAGENT_HF_ENFORCE_EAGER", "0").lower() in (
                    "1", "true", "yes"
                )
                swap_space_gb = int(os.getenv("FOAMAGENT_HF_SWAP_SPACE_GB", "4"))
                print(
                    f"  [gemma4] building vLLM engine: model={model} "
                    f"tokenizer={patched_dir} tp={tensor_parallel_size} "
                    f"max_model_len={max_model_len} lora={need_lora}"
                )
                llm_kwargs = dict(
                    model=model,
                    tokenizer=patched_dir,
                    trust_remote_code=True,
                    tensor_parallel_size=int(tensor_parallel_size),
                    dtype=dtype,
                    quantization=quantization,
                    max_model_len=max_model_len,
                    gpu_memory_utilization=gpu_mem_util,
                    max_num_seqs=max_num_seqs,
                    enforce_eager=enforce_eager,
                    swap_space=swap_space_gb,
                )
                if need_lora:
                    max_lora_rank = int(os.getenv("FOAMAGENT_HF_MAX_LORA_RANK", "16"))
                    max_loras = int(os.getenv("FOAMAGENT_HF_MAX_LORAS", "9"))
                    llm_kwargs.update(
                        enable_lora=True,
                        max_loras=max_loras,
                        max_lora_rank=max_lora_rank,
                    )
                llm = LLM(**llm_kwargs)
                _VLLM_MODEL_CACHE[cache_key] = llm
            self._llm = llm
            lock = _VLLM_LOCKS.get(cache_key)
            if lock is None:
                lock = threading.Lock()
                _VLLM_LOCKS[cache_key] = lock
            self._lock = lock

        self._tl = threading.local()

    # ---- usage accounting (same shape as _VLLMChatWrapper) ----
    def _reset_usage(self) -> None:
        self._tl.cached_prompt_tokens = 0
        self._tl.engine_prompt_tokens = 0
        self._tl.engine_completion_tokens = 0

    def _add_usage(self, cached: int, prompt: int, completion: int) -> None:
        if not hasattr(self._tl, "cached_prompt_tokens"):
            self._reset_usage()
        self._tl.cached_prompt_tokens += int(cached or 0)
        self._tl.engine_prompt_tokens += int(prompt or 0)
        self._tl.engine_completion_tokens += int(completion or 0)

    def pop_last_usage(self) -> dict:
        usage = {
            "cached_prompt_tokens": getattr(self._tl, "cached_prompt_tokens", 0),
            "engine_prompt_tokens": getattr(self._tl, "engine_prompt_tokens", 0),
            "engine_completion_tokens": getattr(self._tl, "engine_completion_tokens", 0),
        }
        self._reset_usage()
        return usage

    # ---- chat template + tokenization ----
    def _render_prompt(self, messages) -> str:
        return self._chat_template.render(
            messages=list(messages),
            add_generation_prompt=True,
            bos_token=self._bos_token,
            eos_token=self._eos_token,
            enable_thinking=False,
        )

    def get_num_tokens(self, text: str) -> int:
        return len(self._tok.encode(text or "", add_special_tokens=False))

    # _strip_reasoning_channels and _extract_json_object reused via the static
    # methods on the generic wrapper.
    _strip_reasoning_channels = staticmethod(_VLLMChatWrapper._strip_reasoning_channels)
    _extract_json_object = staticmethod(_VLLMChatWrapper._extract_json_object)

    def with_structured_output(self, pydantic_obj: Type[BaseModel]):
        parent = self

        class _StructuredWrapper:
            def get_num_tokens(self, text: str) -> int:
                return parent.get_num_tokens(text)

            def invoke(self, messages):
                schema = pydantic_obj.model_json_schema()
                schema_hint = (
                    "Return ONLY valid JSON (no markdown) that matches this JSON Schema:\n"
                    + str(schema)
                )
                patched = list(messages)
                patched.insert(0, {"role": "system", "content": schema_hint})

                last_raw = ""
                for _ in range(3):
                    resp = parent.invoke(patched)
                    raw = parent._strip_reasoning_channels(getattr(resp, "content", "") or "")
                    last_raw = raw
                    try:
                        json_text = parent._extract_json_object(raw)
                        return pydantic_obj.model_validate_json(json_text)
                    except Exception as e:
                        repair = (
                            "Your previous output was not valid JSON for the requested schema. "
                            "Return ONLY valid JSON (no markdown), with no extra keys. "
                            f"Error: {str(e)}\n\nPrevious output:\n{raw}"
                        )
                        patched = list(patched) + [{"role": "user", "content": repair}]
                raise ValueError(
                    f"Failed to produce valid JSON after retries. Last output: {last_raw[:500]}"
                )

        return _StructuredWrapper()

    def invoke(self, messages):
        prompt = self._render_prompt(messages)

        max_tokens = self._max_tokens
        if self._max_model_len is not None:
            prompt_token_count = len(self._tok.encode(prompt, add_special_tokens=False))
            available = self._max_model_len - prompt_token_count - 64
            if available < max_tokens:
                if available < 256:
                    raise ValueError(
                        f"Prompt is {prompt_token_count} tokens, leaving only {available} for "
                        f"output (max_model_len={self._max_model_len}). Cannot generate."
                    )
                max_tokens = available

        params = self._SamplingParams(
            temperature=self._temperature,
            max_tokens=max_tokens,
        )

        with self._lock:
            if self._lora_request is not None:
                out = self._llm.generate(
                    [prompt], params, lora_request=self._lora_request,
                )[0]
            else:
                out = self._llm.generate([prompt], params)[0]

        cached = int(getattr(out, "num_cached_tokens", 0) or 0)
        eng_prompt = len(getattr(out, "prompt_token_ids", []) or [])
        eng_completion = sum(
            len(getattr(o, "token_ids", []) or [])
            for o in (getattr(out, "outputs", []) or [])
        )
        self._add_usage(cached, eng_prompt, eng_completion)

        text = out.outputs[0].text if getattr(out, "outputs", None) else ""
        text = self._strip_reasoning_channels(text or "")
        return self._Resp(text.strip())


# -------------------------
# TensorRT-LLM via trtllm-serve (HTTP / OpenAI-compatible)
# -------------------------
# Why this exists, given we already have an in-process vLLM wrapper:
#   - vLLM does not support LoRA on Gemma 4 — `vllm/lora/utils.py` rejects
#     `Gemma4ForConditionalGeneration` in the supported-models registry (seen
#     when --enable-lora is set on a Gemma 4 base). The merge-and-load fallback
#     would mean materializing 9 distinct 26B checkpoints, which doesn't fit.
#   - TensorRT-LLM 1.3.0rc13's autodeploy backend DOES register Gemma 4 and
#     supports per-request LoRA mounting via `lora_request` on
#     `/v1/chat/completions`. trtllm-serve loads the base model once, registers
#     up to N adapters, and routes each request to its addressed adapter using
#     a stable lora_int_id mapping printed at startup.
#   - We isolate TRT-LLM into its own venv (`trt-llm-env/`) because its CUDA 13
#     wheel set conflicts with the rest of the agent. Keeping the benchmark in
#     `foam-env/` and talking to TRT-LLM over HTTP avoids merging the envs.
#
# Per-substep selection: each substep config in gemma4_finetuned_models.json
# carries its own (lora_name, lora_int_id, lora_path). The wrapper sends those
# verbatim in `lora_request`. The lora_int_id values must match the order
# `serve_trtllm_lora.sh` enumerated the adapters in (parse_case_info=1, ...).
class _TRTLLMOpenAIChatWrapper:
    """OpenAI-compatible HTTP client for trtllm-serve with per-request LoRA.

    Targets the chat-completions endpoint so trtllm-serve renders the chat
    template server-side using the tokenizer it was launched with (the
    sanitized Gemma 4 tokenizer dir produced by serve_trtllm_lora.sh's
    `_tokenizer_sanitized`). This avoids replicating the chat template in the
    client and keeps prompt formatting identical to the server's view.
    """

    class _Resp:
        def __init__(self, content: str):
            self.content = content

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        lora_name: Optional[str] = None,
        lora_int_id: Optional[int] = None,
        lora_path: Optional[str] = None,
        timeout: float = 600.0,
        normalize_chat_roles: bool = False,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._timeout = float(timeout)
        self._normalize_chat_roles = bool(normalize_chat_roles)
        self._lora_request: Optional[Dict[str, Any]] = None
        if lora_int_id is not None:
            if lora_name is None or lora_path is None:
                raise ValueError(
                    "lora_int_id was set but lora_name or lora_path is missing — "
                    "all three must be present together."
                )
            self._lora_request = {
                "lora_name": str(lora_name),
                "lora_int_id": int(lora_int_id),
                "lora_path": str(lora_path),
            }
        # Token counting falls back to tiktoken when the server doesn't return
        # usage. The exact tokenization differs from Gemma 4's SentencePiece, but
        # the bench only uses these numbers for retry budgeting and reporting.
        # Keep this offline-safe: tiktoken downloads encodings on a cold cache.
        try:
            self._enc = tiktoken.get_encoding("o200k_base")
        except Exception:
            try:
                self._enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._enc = None
        self._tl = threading.local()

    # ---- usage accounting (matches _VLLMChatWrapper interface) ----
    def _reset_usage(self) -> None:
        self._tl.cached_prompt_tokens = 0
        self._tl.engine_prompt_tokens = 0
        self._tl.engine_completion_tokens = 0

    def _add_usage(self, cached: int, prompt: int, completion: int) -> None:
        if not hasattr(self._tl, "cached_prompt_tokens"):
            self._reset_usage()
        self._tl.cached_prompt_tokens += int(cached or 0)
        self._tl.engine_prompt_tokens += int(prompt or 0)
        self._tl.engine_completion_tokens += int(completion or 0)

    def pop_last_usage(self) -> dict:
        usage = {
            "cached_prompt_tokens": getattr(self._tl, "cached_prompt_tokens", 0),
            "engine_prompt_tokens": getattr(self._tl, "engine_prompt_tokens", 0),
            "engine_completion_tokens": getattr(self._tl, "engine_completion_tokens", 0),
        }
        self._reset_usage()
        return usage

    def get_num_tokens(self, text: str) -> int:
        if self._enc is None:
            # Crude but stable offline estimate; only affects reporting/retry
            # budget, not the server-side prompt.
            return max(1, len((text or "").encode("utf-8")) // 4)
        return len(self._enc.encode(text or ""))

    # Reuse the vLLM wrapper's robust JSON object extractor — same problem
    # (model emits JSON with prose around it / inside braces).
    _strip_reasoning_channels = staticmethod(_VLLMChatWrapper._strip_reasoning_channels)
    _extract_json_object = staticmethod(_VLLMChatWrapper._extract_json_object)

    def with_structured_output(self, pydantic_obj: Type[BaseModel]):
        parent = self

        class _StructuredWrapper:
            def get_num_tokens(self, text: str) -> int:
                return parent.get_num_tokens(text)

            def invoke(self, messages):
                schema = pydantic_obj.model_json_schema()
                schema_hint = (
                    "Return ONLY valid JSON (no markdown) that matches this JSON Schema:\n"
                    + str(schema)
                )
                patched = list(messages)
                patched.insert(0, {"role": "system", "content": schema_hint})

                last_raw = ""
                for _ in range(3):
                    resp = parent.invoke(patched)
                    raw = parent._strip_reasoning_channels(getattr(resp, "content", "") or "")
                    last_raw = raw
                    try:
                        json_text = parent._extract_json_object(raw)
                        return pydantic_obj.model_validate_json(json_text)
                    except Exception as e:
                        repair = (
                            "Your previous output was not valid JSON for the requested schema. "
                            "Return ONLY valid JSON (no markdown), with no extra keys. "
                            f"Error: {str(e)}\n\nPrevious output:\n{raw}"
                        )
                        patched = list(patched) + [{"role": "user", "content": repair}]
                raise ValueError(
                    f"Failed to produce valid JSON after retries. Last output: {last_raw[:500]}"
                )

        return _StructuredWrapper()

    @staticmethod
    def _message_content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return str(content)

    @classmethod
    def _append_chat_message(
        cls,
        out: List[Dict[str, str]],
        role: str,
        content: Any,
    ) -> None:
        text = cls._message_content_to_text(content).strip()
        if not text:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"] = f"{out[-1]['content']}\n\n{text}"
            return
        out.append({"role": role, "content": text})

    @classmethod
    def _normalize_for_alternating_chat_template(
        cls,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Convert OpenAI-style messages to user/assistant alternation.

        Gemma 4's served HF chat template rejects `system` turns and repeated
        roles. Keep the instruction text by folding system/developer content
        into the next user turn, then merge adjacent equal roles.
        """
        normalized: List[Dict[str, str]] = []
        pending_system: List[str] = []

        for msg in messages:
            role = str(msg.get("role") or "user").lower()
            content = cls._message_content_to_text(msg.get("content"))

            if role in {"system", "developer"}:
                if content.strip():
                    pending_system.append(content.strip())
                continue

            if role not in {"user", "assistant"}:
                role = "user"

            if pending_system and role == "user":
                content = "\n\n".join(pending_system + ([content] if content.strip() else []))
                pending_system = []
            elif pending_system:
                cls._append_chat_message(normalized, "user", "\n\n".join(pending_system))
                pending_system = []

            cls._append_chat_message(normalized, role, content)

        if pending_system:
            cls._append_chat_message(normalized, "user", "\n\n".join(pending_system))

        if not normalized:
            return [{"role": "user", "content": ""}]

        if normalized[0]["role"] != "user":
            normalized[0] = {
                "role": "user",
                "content": f"Previous assistant message:\n{normalized[0]['content']}",
            }

        collapsed: List[Dict[str, str]] = []
        for msg in normalized:
            cls._append_chat_message(collapsed, msg["role"], msg["content"])
        return collapsed or [{"role": "user", "content": ""}]

    def _build_payload(self, messages):
        outbound_messages = list(messages)
        if self._normalize_chat_roles:
            outbound_messages = self._normalize_for_alternating_chat_template(outbound_messages)

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": outbound_messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        if self._lora_request is not None:
            # trtllm-serve's ChatCompletionRequest accepts a top-level lora_request
            # field (openai_protocol.py:692). It is NOT under extra_body — the
            # field is part of TRT-LLM's extended schema.
            payload["lora_request"] = self._lora_request
        return payload

    def invoke(self, messages):
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Foam-Agent",
        }
        payload = self._build_payload(messages)

        context_retries = int(os.getenv("FOAMAGENT_HTTP_CONTEXT_RETRIES", "0"))
        min_context_output = int(os.getenv("FOAMAGENT_HTTP_CONTEXT_MIN_OUTPUT_TOKENS", "512"))
        context_attempt = 0
        while True:
            r = requests.post(url, headers=headers, json=payload, timeout=self._timeout)
            if r.ok:
                break
            try:
                detail = r.text[:2000]
            except Exception:
                detail = ""

            current_max_tokens = int(payload.get("max_tokens") or self._max_tokens)
            next_max_tokens = max(min_context_output, current_max_tokens // 2)
            is_context_limit = (
                r.status_code == 400
                and "maximum context length" in detail.lower()
            )
            if (
                is_context_limit
                and context_attempt < context_retries
                and next_max_tokens < current_max_tokens
            ):
                context_attempt += 1
                payload["max_tokens"] = next_max_tokens
                print(
                    "HTTP context retry "
                    f"{context_attempt}/{context_retries}: max_tokens "
                    f"{current_max_tokens} -> {next_max_tokens}"
                )
                continue

            raise requests.HTTPError(
                f"HTTP {r.status_code} for {url}. Body: {detail}", response=r
            )
        data = r.json()

        # Standard OpenAI shape: {"choices": [{"message": {"content": "..."}}], "usage": {...}}
        choices = data.get("choices") or []
        text = ""
        if choices:
            msg = choices[0].get("message") or {}
            text = msg.get("content") or ""

        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        # cached_tokens may live under prompt_tokens_details for OpenAI parity.
        cached_tokens = 0
        ptd = usage.get("prompt_tokens_details") or {}
        if isinstance(ptd, dict):
            cached_tokens = int(ptd.get("cached_tokens") or 0)
        self._add_usage(cached_tokens, prompt_tokens, completion_tokens)

        text = self._strip_reasoning_channels(text or "")
        return self._Resp(text.strip())


class LLMService:
    @staticmethod
    def _load_codex_access_token_from_auth_json(auth_json_path: Path) -> str:
        import json

        data = json.loads(auth_json_path.read_text(encoding="utf-8"))

        # Be permissive: different Codex versions may store different shapes.
        # Common patterns we try:
        #   {"access_token": "..."}
        #   {"token": "..."}
        #   {"auth": {"access_token": "..."}}
        #   {"credentials": {"access_token": "..."}}
        #   {"tokens": {"access_token": "...", "account_id": "..."}}   (current Codex CLI)
        candidates = []

        def maybe_add(v):
            if isinstance(v, str) and v.strip():
                candidates.append(v.strip())

        if isinstance(data, dict):
            maybe_add(data.get("access_token"))
            maybe_add(data.get("token"))

            for k in ("tokens", "auth", "credentials", "session"):
                v = data.get(k)
                if isinstance(v, dict):
                    maybe_add(v.get("access_token"))
                    maybe_add(v.get("token"))

        if not candidates:
            raise ValueError(
                f"Could not find an access token in {auth_json_path}. "
                "Expected keys like access_token/token/id_token."
            )

        # Prefer access_token-like strings first (we already appended in that order)
        return candidates[0]

    @staticmethod
    def _load_codex_oauth_from_clawdbot_auth_profiles(auth_profiles_path: Path) -> tuple[str, Optional[str]]:
        """Load (access token, account id) from Clawdbot's OpenAI-Codex OAuth cache.

        Expected shape (v1):
          {"profiles": {"openai-codex:default": {"access": "...", "accountId": "...", ...}}}

        We also fall back to "openai-codex" or any first profile that looks usable.
        """
        import json

        data = json.loads(auth_profiles_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected JSON in {auth_profiles_path}")

        profiles = data.get("profiles")
        if not isinstance(profiles, dict):
            raise ValueError(f"Missing 'profiles' in {auth_profiles_path}")

        preferred_keys = ["openai-codex:default", "openai-codex"]
        for k in preferred_keys:
            v = profiles.get(k)
            if isinstance(v, dict):
                token = v.get("access")
                account_id = v.get("accountId")
                if isinstance(token, str) and token.strip():
                    return token.strip(), account_id if isinstance(account_id, str) else None

        # Fallback: scan any profile entry that has an 'access' string
        for _, v in profiles.items():
            if isinstance(v, dict):
                token = v.get("access")
                account_id = v.get("accountId")
                if isinstance(token, str) and token.strip():
                    return token.strip(), account_id if isinstance(account_id, str) else None

        raise ValueError(
            f"Could not find an 'access' token in {auth_profiles_path}. "
            "Expected profiles[*].access"
        )

    def _load_codex_oauth(self) -> tuple[str, Optional[str]]:
        """Load the Codex/ChatGPT OAuth token from a local auth cache.

        Supported locations (first match wins):
        1) $CODEX_HOME/auth.json (Codex CLI file-based cache)
        2) ~/.codex/auth.json (Codex CLI default)
        3) ~/.clawdbot/agents/main/agent/auth-profiles.json (Clawdbot OpenAI-Codex OAuth cache)

        Note: These files contain access/refresh tokens. Treat them like passwords.
        """
        candidates: list[Path] = []

        codex_home = os.getenv("CODEX_HOME")
        if codex_home:
            candidates.append(Path(codex_home) / "auth.json")

        candidates.append(Path.home() / ".codex" / "auth.json")

        # Clawdbot stores the OpenAI-Codex OAuth profile here.
        candidates.append(
            Path.home()
            / ".clawdbot"
            / "agents"
            / "main"
            / "agent"
            / "auth-profiles.json"
        )

        for p in candidates:
            if not p.exists():
                continue

            # Codex CLI cache
            if p.name == "auth.json":
                import json
                token = self._load_codex_access_token_from_auth_json(p)
                # The current Codex CLI keeps the ChatGPT account id next to the token; send it as ChatGPT-Account-Id.
                account_id = None
                try:
                    tokens = json.loads(p.read_text(encoding="utf-8")).get("tokens")
                    if isinstance(tokens, dict) and isinstance(tokens.get("account_id"), str):
                        account_id = tokens["account_id"]
                except Exception:
                    pass
                return token, account_id

            # Clawdbot cache
            if p.name == "auth-profiles.json":
                return self._load_codex_oauth_from_clawdbot_auth_profiles(p)

        raise FileNotFoundError(
            "Could not find a Codex/ChatGPT OAuth cache. Looked for: "
            + ", ".join(str(x) for x in candidates)
            + ". "
            "If you used the Codex CLI, run `codex login` and ensure file-based credential storage. "
            "If you used Clawdbot, make sure you completed OpenAI Codex OAuth in onboarding."
        )

    def __init__(self, config: object):
        self.models: Dict[str, Dict[str, Union[str, float]]] = getattr(config, "models")
        self.services = self.models.keys()
        self.llms: Dict[str, Any] = {}
        self._config = config

        # Initialize statistics
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        # Engine-reported token accounting (currently populated by the vLLM
        # wrapper; zero for providers that don't surface cache stats). Lets
        # the summarizer report prefix-cache hit rate.
        self.total_cached_prompt_tokens = 0
        self.total_engine_prompt_tokens = 0
        self.total_engine_completion_tokens = 0
        self.failed_calls = 0
        self.retry_count = 0

        # Dataset logging
        self.dataset_log_path = getattr(config, "dataset_log_path", "") or ""
        self.case_id = getattr(config, "case_id", "") or ""
        self._dataset_log_lock = threading.Lock()

        for service in self.services:
            self.llms[service] = self._instantiate_llm(self.models[service])

        # Substep-level overrides: keyed by substep name, take priority over service-level in invoke()
        self.substep_llms:   Dict[str, Any]            = {}
        self.substep_models: Dict[str, Dict[str, Any]] = {}
        substep_overrides = getattr(config, "substep_model_overrides", {}) or {}
        for substep_name, model_cfg in substep_overrides.items():
            self.substep_llms[substep_name]   = self._instantiate_llm(model_cfg)
            self.substep_models[substep_name] = model_cfg
    
    def _instantiate_llm(self, model_cfg: dict) -> Any:
        """Instantiate a single LLM from a model config dict."""
        model_provider = model_cfg["model_provider"]
        model_version  = model_cfg["model_version"]
        temperature    = model_cfg["temperature"]

        if model_provider.lower() == "bedrock":
            bedrock_runtime = tracking_aws.new_default_client()
            return ChatBedrockConverse(
                client=bedrock_runtime,
                model_id=model_version,
                temperature=temperature,
                max_tokens=8192
            )
        elif model_provider.lower() == "anthropic":
            return ChatAnthropic(
                model=model_version,
                temperature=temperature
            )
        elif model_provider.lower() == "openai":
            # Usage-based API access (requires OPENAI_API_KEY or equivalent OpenAI SDK config)
            return init_chat_model(
                model_version,
                model_provider="openai",
                temperature=temperature,
            )
        elif model_provider.lower() in {"huggingface", "hf"}:
            # Local inference via vLLM. model_version is a HuggingFace repo id or a local path.
            # Optionally supports LoRA adapters via lora_adapter_path in the model config.
            tp = int(os.getenv("FOAMAGENT_HF_TP", "1"))
            dtype = os.getenv("FOAMAGENT_HF_DTYPE", "bfloat16")
            quant = os.getenv("FOAMAGENT_HF_QUANT")
            max_len_env = os.getenv("FOAMAGENT_HF_MAX_MODEL_LEN")
            max_model_len = int(max_len_env) if max_len_env else None
            # Reasoning models (GPT-OSS, DeepSeek-R1, etc.) burn output tokens
            # on internal reasoning before producing the final answer, so 8k can
            # truncate mid-JSON. Default to 16k; override via env.
            max_tokens = int(os.getenv("FOAMAGENT_HF_MAX_OUTPUT_TOKENS", "16384"))
            lora_path = model_cfg.get("lora_adapter_path")
            chat_template_path = model_cfg.get("chat_template_path")
            tokenizer_path = model_cfg.get("tokenizer_path")
            model_type = (model_cfg.get("model_type") or "").lower()
            if model_type == "gemma4":
                # Specialized path: Gemma 4's tokenizer_config.json shape
                # breaks AutoTokenizer in this transformers version, and the
                # native chat template diverges from the unsloth `gemma-4`
                # alias used at training time.
                return _VLLMGemma4ChatWrapper(
                    model=model_version,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tensor_parallel_size=tp,
                    dtype=dtype,
                    quantization=quant,
                    max_model_len=max_model_len,
                    lora_adapter_path=lora_path,
                    chat_template_path=chat_template_path,
                    tokenizer_path=tokenizer_path,
                )
            return _VLLMChatWrapper(
                model=model_version,
                temperature=temperature,
                max_tokens=max_tokens,
                tensor_parallel_size=tp,
                dtype=dtype,
                quantization=quant,
                max_model_len=max_model_len,
                lora_adapter_path=lora_path,
                chat_template_path=chat_template_path,
            )
        elif model_provider.lower() in {"trtllm-openai", "trtllm", "trt-llm",
                                         "vllm-openai", "vllm-http"}:
            # OpenAI-compatible HTTP client. Works with any server that speaks
            # the OpenAI Chat Completions wire format:
            #   - trtllm-serve (./serve_trtllm_lora.sh, trt-llm-env venv) —
            #     base+LoRA path; uses the `lora_request` extension field.
            #   - vllm serve (./serve_vllm_merged.sh + launch_vllm_fleet.sbatch,
            #     foam-env venv) — merged-checkpoint path, one process per
            #     substep, lora fields omitted.
            # The benchmark stays in foam-env and only talks to it over HTTP,
            # so the same wrapper works for both.
            base_url = (
                model_cfg.get("base_url")
                or os.getenv("FOAMAGENT_TRTLLM_BASE_URL")
                or "http://localhost:8000/v1"
            )
            timeout = float(model_cfg.get("timeout") or os.getenv("FOAMAGENT_TRTLLM_TIMEOUT", "600"))
            max_tokens = int(
                model_cfg.get("max_tokens")
                or os.getenv("FOAMAGENT_HF_MAX_OUTPUT_TOKENS", "16384")
            )
            return _TRTLLMOpenAIChatWrapper(
                base_url=base_url,
                model=model_version,
                temperature=temperature,
                max_tokens=max_tokens,
                lora_name=model_cfg.get("lora_name"),
                lora_int_id=model_cfg.get("lora_int_id"),
                lora_path=model_cfg.get("lora_path") or model_cfg.get("lora_adapter_path"),
                timeout=timeout,
                normalize_chat_roles=model_provider.lower() in {"vllm-openai", "vllm-http"}
                or bool(model_cfg.get("normalize_chat_roles")),
            )
        elif model_provider.lower() in {"openai-codex", "codex", "chatgpt-oauth"}:
            # Subscription-based access via "Sign in with ChatGPT" (Codex auth cache).
            token, account_id = self._load_codex_oauth()
            instructions_path = Path(__file__).resolve().parent / "codex_instructions_default.txt"
            try:
                instructions = instructions_path.read_text(encoding="utf-8")
            except Exception:
                instructions = "You are Codex, based on GPT-5. You are running as a coding agent in the Codex CLI on a user's computer."
            return _CodexResponsesWrapper(
                token=token,
                account_id=account_id,
                model=model_version,
                temperature=temperature,
                base_url="https://chatgpt.com/backend-api/codex",
                instructions=instructions,
                stream=True,
            )
        elif model_provider.lower() == "ollama":
            try:
                requests.get("http://localhost:11434/api/version", timeout=2)
                # If request successful, service is running
            except requests.exceptions.RequestException:
                print("Ollama is not running, starting it...")
                subprocess.Popen(["ollama", "serve"],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
                # Wait for service to start
                time.sleep(5)
            return ChatOllama(
                model=model_version,
                temperature=temperature,
                num_predict=-1,
                num_ctx=131072,
                base_url="http://localhost:11434"
            )
        else:
            raise ValueError(f"{model_provider} is not a supported model_provider")

    def _is_throttling_error(self, error: Exception) -> bool:
        """
        Check if an exception is a throttling-related error.
        
        Args:
            error: The exception to check
            
        Returns:
            True if it's a throttling error, False otherwise
        """
        # Check ClientError with specific error codes
        if isinstance(error, ClientError):
            error_code = error.response.get('Error', {}).get('Code', '')
            return error_code in ('Throttling', 'TooManyRequestsException', 'ThrottlingException')
        
        # Check for ThrottlingException and throttling-related error messages
        error_type = type(error).__name__
        error_str = str(error)
        
        throttling_indicators = (
            error_type == 'ThrottlingException',
            'ThrottlingException' in error_str,
            'Too many tokens' in error_str,
            'reached max retries' in error_str
        )
        
        return any(throttling_indicators)
    
    def _handle_throttling_retry(self, error: Exception, retry_count: int, max_retries: int) -> Optional[int]:
        """
        Handle throttling error by implementing exponential backoff retry logic.
        
        Args:
            error: The throttling exception
            retry_count: Current retry attempt number
            max_retries: Maximum number of retries allowed
            
        Returns:
            The updated retry count if retry should continue, None if max retries exceeded
        """
        retry_count += 1
        self.retry_count += 1
        
        if retry_count > max_retries:

            print(f"Maximum retries ({max_retries}) exceeded: {str(error)}")
            return None
        
        # Exponential backoff with jitter
        base_delay = 1.0
        max_delay = 60.0
        delay = min(max_delay, base_delay * (2 ** (retry_count - 1)))
        jitter = random.uniform(0, 0.1 * delay)
        sleep_time = delay + jitter
        
        print(f"ThrottlingException occurred: {str(error)}. Retrying in {sleep_time:.2f} seconds (attempt {retry_count}/{max_retries})")
        time.sleep(sleep_time)
        
        return retry_count
    
    def _log_dataset_record(self, log_context: dict, system_prompt: str,
                            user_prompt: str, response: str, service: str,
                            prompt_tokens: int, completion_tokens: int,
                            model_version: str = "") -> None:
        """Append a JSONL record for fine-tuning dataset extraction (thread-safe)."""
        import json as _json
        from datetime import datetime, timezone
        record = {
            "case_id": self.case_id,
            "step": log_context.get("step", ""),
            "substep": log_context.get("substep", ""),
            "loop_iteration": log_context.get("loop_iteration", 0),
            "file_target": log_context.get("file_target", ""),
            "system_prompt": system_prompt or "",
            "user_prompt": user_prompt,
            "response": response,
            "model": model_version or self.models[service]["model_version"],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self._dataset_log_lock:
            os.makedirs(os.path.dirname(self.dataset_log_path), exist_ok=True)
            with open(self.dataset_log_path, "a") as f:
                f.write(_json.dumps(record, ensure_ascii=False) + "\n")

    def invoke(self,
               service: str,
               user_prompt: str,
               system_prompt: Optional[str] = None,
               pydantic_obj: Optional[Type[BaseModel]] = None,
               max_retries: int = 10,
               log_context: Optional[dict] = None) -> Any:
        """
        Invoke the LLM with the given prompts and return the response.
        
        Args:
            service: plan, write, review
            user_prompt: The user's prompt
            system_prompt: Optional system prompt
            pydantic_obj: Optional Pydantic model for structured output
            max_retries: Maximum number of retries for throttling errors
            
        Returns:
            The LLM response with token usage statistics
        """
        self.total_calls += 1

        # Resolve per-substep override (substep takes priority over service-level)
        substep = (log_context or {}).get("substep", "")
        # kit_e49: per-call context for the call log (call site, case log dir)
        _CALL_CTX.substep = substep
        _CALL_CTX.log_dir = os.path.dirname(self.dataset_log_path) if getattr(self, "dataset_log_path", None) else None
        if substep and substep in self.substep_llms:
            effective_llm   = self.substep_llms[substep]
            effective_model = self.substep_models[substep]
        else:
            effective_llm   = self.llms[service]
            effective_model = self.models[service]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # Calculate prompt tokens
        prompt_tokens = 0
        for message in messages:
            prompt_tokens += effective_llm.get_num_tokens(message["content"])

        retry_count = 0
        while True:
            try:
                # Clear the wrapper's per-thread usage accumulator before the
                # call so whatever we read afterwards belongs to this invoke
                # (vLLM path only; other providers no-op).
                pop_usage = getattr(effective_llm, "pop_last_usage", None)
                if pop_usage is not None:
                    pop_usage()

                if pydantic_obj:

                    structured_llm = effective_llm.with_structured_output(pydantic_obj)
                    response = structured_llm.invoke(messages)
                else:
                    if effective_model["model_version"].startswith("deepseek"):
                        structured_llm = effective_llm.with_structured_output(ResponseWithThinkPydantic)
                        response = structured_llm.invoke(messages)

                        # Extract the resposne without the think
                        response = response.response
                    else:
                        response = effective_llm.invoke(messages)
                        response = response.content

                # Strip thinking/reasoning tags from models like DeepSeek, QwQ, Nemotron, etc.
                if isinstance(response, str):
                    response = re.sub(r"<(?:think|thinking|reasoning)>.*?</(?:think|thinking|reasoning)>\s*", "", response, flags=re.DOTALL).strip()

                # Calculate completion tokens
                response_content = str(response)
                completion_tokens = effective_llm.get_num_tokens(response_content)
                total_tokens = prompt_tokens + completion_tokens

                # Pull the engine's own accounting (cached / prompt / completion
                # token counts) if the wrapper populated it. Summed across any
                # structured-output retry attempts inside the wrapper.
                engine_usage = pop_usage() if pop_usage is not None else {}
                cached_prompt_tokens = int(engine_usage.get("cached_prompt_tokens", 0))
                engine_prompt_tokens = int(engine_usage.get("engine_prompt_tokens", 0))
                engine_completion_tokens = int(engine_usage.get("engine_completion_tokens", 0))

                # Update statistics
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_tokens += total_tokens
                self.total_cached_prompt_tokens += cached_prompt_tokens
                self.total_engine_prompt_tokens += engine_prompt_tokens
                self.total_engine_completion_tokens += engine_completion_tokens

                print(f"*** service={service}, model_version={effective_model['model_version']} ***")

                # Dataset logging
                if log_context and self.dataset_log_path:
                    self._log_dataset_record(
                        log_context, system_prompt, user_prompt,
                        response_content, service, prompt_tokens, completion_tokens,
                        model_version=effective_model["model_version"],
                    )

                return response
                
            except Exception as e:
                if self._is_throttling_error(e):
                    print(f"ThrottlingException occurred: {str(e)}.")
                    print(f"Retrying: {retry_count + 1}/{max_retries}")
                    retry_count = self._handle_throttling_retry(e, retry_count, max_retries)
                    if retry_count is None:
                        # Max retries exceeded
                        self.failed_calls += 1
                        raise Exception(f"Maximum retries ({max_retries}) exceeded for throttling error: {str(e)}")
                    continue  # Retry the request
                else:
                    print(f"Non-throttling error occurred: {str(e)}.")

                    # Non-throttling error: log and raise
                    print(f"Error occurred in LLM service: {str(e)}")
                    if isinstance(e, ClientError):
                        print(e.response)
                    self.failed_calls += 1
                    raise e
    
    def get_statistics(self) -> dict:
        """
        Get the current statistics of the LLM service.
        
        Returns:
            Dictionary containing various statistics
        """
        cache_denom = self.total_engine_prompt_tokens
        cache_hit_rate = (
            self.total_cached_prompt_tokens / cache_denom if cache_denom > 0 else 0.0
        )
        return {
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "retry_count": self.retry_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cached_prompt_tokens": self.total_cached_prompt_tokens,
            "total_engine_prompt_tokens": self.total_engine_prompt_tokens,
            "total_engine_completion_tokens": self.total_engine_completion_tokens,
            "cache_hit_rate": cache_hit_rate,
            "average_prompt_tokens": self.total_prompt_tokens / self.total_calls if self.total_calls > 0 else 0,
            "average_completion_tokens": self.total_completion_tokens / self.total_calls if self.total_calls > 0 else 0,
            "average_tokens": self.total_tokens / self.total_calls if self.total_calls > 0 else 0
        }
    
    def print_statistics(self) -> None:
        """
        Print the current statistics of the LLM service.
        """
        stats = self.get_statistics()
        print("\n<LLM Service Statistics>")
        print(f"Total calls: {stats['total_calls']}")
        print(f"Failed calls: {stats['failed_calls']}")
        print(f"Total retries: {stats['retry_count']}")
        print(f"Total prompt tokens: {stats['total_prompt_tokens']}")
        print(f"Total completion tokens: {stats['total_completion_tokens']}")
        print(f"Total tokens: {stats['total_tokens']}")
        # Engine-reported accounting (vLLM prefix cache). Zero for providers
        # that don't report it. engine_prompt_tokens reflects the rendered
        # chat-template prompt vLLM actually saw, which may differ from the
        # content-only total_prompt_tokens above.
        print(f"Total cached prompt tokens: {stats['total_cached_prompt_tokens']}")
        print(f"Total engine prompt tokens: {stats['total_engine_prompt_tokens']}")
        print(f"Total engine completion tokens: {stats['total_engine_completion_tokens']}")
        print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
        print(f"Average prompt tokens per call: {stats['average_prompt_tokens']:.2f}")
        print(f"Average completion tokens per call: {stats['average_completion_tokens']:.2f}")
        print(f"Average tokens per call: {stats['average_tokens']:.2f}\n")
        print("</LLM Service Statistics>")

class GraphState(TypedDict):
    user_requirement: str
    config: Config
    case_dir: str
    tutorial: str
    case_name: str
    subtasks: List[dict]
    current_subtask_index: int
    error_command: Optional[str]
    error_content: Optional[str]
    loop_count: int
    # Additional state fields that will be added during execution
    llm_service: Optional['LLMService']
    case_stats: Optional[dict]
    tutorial_reference: Optional[str]
    case_path_reference: Optional[str]
    dir_structure_reference: Optional[str]
    case_info: Optional[str]
    allrun_reference: Optional[str]
    dir_structure: Optional[dict]
    commands: Optional[List[str]]
    foamfiles: Optional[dict]
    error_logs: Optional[List[str]]
    history_text: Optional[List[str]]
    case_domain: Optional[str]
    case_category: Optional[str]
    case_solver: Optional[str]
    # Mesh-related state fields
    mesh_info: Optional[dict]
    mesh_commands: Optional[List[str]]
    custom_mesh_used: Optional[bool]
    mesh_type: Optional[str]
    custom_mesh_path: Optional[str]
    # Review and rewrite related fields
    review_analysis: Optional[str]
    rewrite_plan: Optional[dict]
    input_writer_mode: Optional[str]
    similar_case_advice: Optional[dict]
    # Routing decision cache
    requires_hpc: Optional[bool]
    requires_visualization: Optional[bool]
    # HPC-related fields
    job_id: Optional[str]
    cluster_info: Optional[dict]
    slurm_script_path: Optional[str]
    termination_reason: Optional[str]

def tokenize(text: str) -> str:
    # Replace underscores with spaces
    text = text.replace('_', ' ')
    # Insert a space between a lowercase letter and an uppercase letter (global match)
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    return text.lower()

def save_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    print(f"Saved file at {path}")

def read_file(path: str) -> str:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return ""

def list_case_files(case_dir: str) -> str:
    files = [f for f in os.listdir(case_dir) if os.path.isfile(os.path.join(case_dir, f))]
    return ", ".join(files)

def remove_files(directory: str, prefix: str) -> None:
    for file in os.listdir(directory):
        if file.startswith(prefix):
            os.remove(os.path.join(directory, file))
    print(f"Removed files with prefix '{prefix}' in {directory}")

def remove_file(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
        print(f"Removed file {path}")

def remove_numeric_folders(case_dir: str) -> None:
    """
    Remove all folders in case_dir that represent numeric values, including those with decimal points,
    except for the "0" folder.
    
    Args:
        case_dir (str): The directory path to process
    """
    for item in os.listdir(case_dir):
        item_path = os.path.join(case_dir, item)
        if os.path.isdir(item_path) and item != "0":
            try:
                # Try to convert to float to check if it's a numeric value
                float(item)
                # If conversion succeeds, it's a numeric folder
                try:
                    shutil.rmtree(item_path)
                    print(f"Removed numeric folder: {item_path}")
                except Exception as e:
                    print(f"Error removing folder {item_path}: {str(e)}")
            except ValueError:
                # Not a numeric value, so we keep this folder
                pass


def scan_case_directory(case_dir: str) -> Dict[str, List[str]]:
    """
    Scan an OpenFOAM case directory and return the directory structure.
    
    This function traverses the case directory one level deep and collects
    the files in each subdirectory (typically 'system', 'constant', '0', etc.).
    
    Args:
        case_dir (str): Path to the OpenFOAM case directory
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping folder names to lists of file names
            Example: {"system": ["controlDict", "fvSchemes"], "constant": ["transportProperties"]}
    
    Raises:
        FileNotFoundError: If case_dir does not exist
        PermissionError: If directory cannot be accessed
    
    Example:
        >>> structure = scan_case_directory("/path/to/case")
        >>> print(structure["system"])  # ["controlDict", "fvSchemes", "fvSolution"]
    """
    if not os.path.exists(case_dir):
        raise FileNotFoundError(f"Case directory does not exist: {case_dir}")
    
    dir_structure = {}
    base_depth = case_dir.rstrip(os.sep).count(os.sep)
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(case_dir):
        # Only process directories one level below case_dir
        current_depth = root.rstrip(os.sep).count(os.sep)
        if current_depth == base_depth + 1:
            folder_name = os.path.relpath(root, case_dir)
            # Filter out hidden files and only include regular files
            regular_files = [f for f in files if not f.startswith('.') and os.path.isfile(os.path.join(root, f))]
            if regular_files:
                dir_structure[folder_name] = regular_files
    
    return dir_structure


def read_case_foamfiles(case_dir: str, dir_structure: Optional[Dict[str, List[str]]] = None) -> 'FoamPydantic':
    """
    Read OpenFOAM files from a case directory and convert to FoamPydantic format.
    
    This function reads all OpenFOAM configuration files from the case directory
    (typically from 'system', 'constant', '0' folders) and creates a FoamPydantic
    object containing the file contents.
    
    Args:
        case_dir (str): Path to the OpenFOAM case directory
        dir_structure (Optional[Dict[str, List[str]]]): Pre-scanned directory structure.
            If None, will scan the directory automatically.
    
    Returns:
        FoamPydantic: Object containing list of FoamfilePydantic objects with file metadata
    
    Raises:
        FileNotFoundError: If case_dir does not exist
        UnicodeDecodeError: If files contain invalid encoding (will skip those files)
    
    Example:
        >>> foamfiles = read_case_foamfiles("/path/to/case")
        >>> print(len(foamfiles.list_foamfile))  # Number of files read
        >>> print(foamfiles.list_foamfile[0].file_name)  # "controlDict"
    """
    if not os.path.exists(case_dir):
        raise FileNotFoundError(f"Case directory does not exist: {case_dir}")
    
    # Scan directory structure if not provided
    if dir_structure is None:
        dir_structure = scan_case_directory(case_dir)
    
    foamfile_list = []
    
    # Read files from each folder
    for folder_name, file_names in dir_structure.items():
        for file_name in file_names:
            file_path = os.path.join(case_dir, folder_name, file_name)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                foamfile_list.append(FoamfilePydantic(
                    file_name=file_name,
                    folder_name=folder_name,
                    content=content
                ))
            except UnicodeDecodeError:
                print(f"Warning: Skipping file due to encoding error: {file_path}")
            except Exception as e:
                print(f"Warning: Error reading file {file_path}: {e}")
    
    return FoamPydantic(list_foamfile=foamfile_list)

def run_command(script_path: str, out_file: str, err_file: str, working_dir: str, max_time_limit: int) -> None:
    print(f"Executing script {script_path} in {working_dir}")
    os.chmod(script_path, 0o777)
    openfoam_dir = os.getenv("WM_PROJECT_DIR")
    if not openfoam_dir:
        raise RuntimeError(
            "WM_PROJECT_DIR is not set. Please source OpenFOAM environment before running Foam-Agent "
            "(e.g., source env/common.sh and env/foamagent.sh)."
        )

    bashrc_path = os.path.join(openfoam_dir, "etc", "bashrc")
    if not os.path.exists(bashrc_path):
        raise RuntimeError(f"OpenFOAM bashrc not found at: {bashrc_path}")

    command = f"source {bashrc_path} && bash {os.path.abspath(script_path)}"

    with open(out_file, 'w') as out, open(err_file, 'w') as err:
        process = subprocess.Popen(
            ['bash', "-c", command],
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            start_new_session=True
        )

        try:
            stdout, stderr = process.communicate(timeout=max_time_limit)
            out.write(stdout)
            err.write(stderr)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL) 
            
            stdout, stderr = process.communicate()
            timeout_message = (
                "OpenFOAM execution took too long. "
                "This case, if set up right, does not require such large execution times.\n"
            )
            out.write(timeout_message + stdout)
            err.write(timeout_message + stderr)
            print(f"Execution timed out: {script_path}")

    

    print(f"Executed script {script_path}")

def check_foam_errors(directory: str) -> list:
    """Check OpenFOAM log files for errors.

    Tier 0 (new): Check Allrun.err for shell-level crashes (segfault, command
    not found, syntax error) that prevent any solver from running.
    Tier 1 (existing): Match explicit ``ERROR:`` lines in log files.
    Tier 2 (safety-net): If no explicit error is found, verify that **every**
    log file contains the ``End`` marker that OpenFOAM prints on successful
    completion.  Any log missing ``End`` is reported with the last 30 lines
    as error context so the caller can diagnose the crash.
    Tier 3 (new): No log files at all — the Allrun likely crashed before any
    solver could run.
    """
    error_logs = []
    log_contents = {}  # filename -> content

    # Tier 0: Check Allrun.err for shell-level errors
    err_file = os.path.join(directory, "Allrun.err")
    if os.path.exists(err_file):
        try:
            with open(err_file, 'r') as f:
                err_content = f.read().strip()
            if err_content:
                # Check for common shell-level crash indicators
                shell_error_patterns = [
                    "Segmentation fault", "command not found",
                    "syntax error", "Permission denied",
                    "No such file or directory", "unexpected EOF",
                ]
                for pat in shell_error_patterns:
                    if pat in err_content:
                        error_logs.append({
                            "file": "Allrun.err",
                            "error_content": f"Shell-level error in Allrun: {err_content[-2000:]}"
                        })
                        break
        except (IOError, OSError):
            pass

    # DOTALL mode allows '.' to match newline characters
    pattern = re.compile(r"ERROR:(.*)", re.DOTALL)

    for file in os.listdir(directory):
        if file.startswith("log"):
            filepath = os.path.join(directory, file)
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
            except (IOError, OSError):
                error_logs.append({"file": file, "error_content": f"Could not read log file: {filepath}"})
                continue

            log_contents[file] = content

            match = pattern.search(content)
            if match:
                error_content = match.group(0).strip()
                error_logs.append({"file": file, "error_content": error_content})
            elif "error" in content.lower():
                print(f"Warning: file {file} contains 'error' but does not match expected format.")

    # Tier 3: No log files at all — Allrun likely crashed before any solver ran
    if not error_logs and not log_contents:
        error_logs.append({
            "file": "N/A",
            "error_content": (
                "No OpenFOAM log files found in the run directory. "
                "The Allrun script likely crashed before any solver could execute."
            ),
        })

    # Safety-net: if no explicit ERROR was found, check for missing 'End' marker
    # Check EACH log individually – a successful blockMesh should not mask a
    # crashed solver (e.g. pimpleFoam).
    if not error_logs and log_contents:
        end_pattern = re.compile(r"^\s*End\s*$", re.MULTILINE)

        for file, content in log_contents.items():
            if not end_pattern.search(content):
                last_lines = "\n".join(content.strip().split("\n")[-30:])
                error_logs.append({
                    "file": file,
                    "error_content": (
                        f"Solver did not complete (no 'End' marker found). "
                        f"Last 30 lines:\n{last_lines}"
                    ),
                })

    return error_logs

def extract_commands_from_allrun_out(out_file: str) -> list:
    commands = []
    if not os.path.exists(out_file):
        return commands
    with open(out_file, 'r') as f:
        for line in f:
            if line.startswith("Running "):
                parts = line.split(" ")
                if len(parts) > 1:
                    commands.append(parts[1].strip())
    return commands

def parse_case_name(text: str) -> str:
    match = re.search(r'case name:\s*(.+)', text, re.IGNORECASE)
    return match.group(1).strip() if match else "default_case"

def split_subtasks(text: str) -> list:
    header_match = re.search(r'splits into (\d+) subtasks:', text, re.IGNORECASE)
    if not header_match:
        print("Warning: No subtasks header found in the response.")
        return []
    num_subtasks = int(header_match.group(1))
    subtasks = re.findall(r'subtask\d+:\s*(.*)', text, re.IGNORECASE)
    if len(subtasks) != num_subtasks:
        print(f"Warning: Expected {num_subtasks} subtasks but found {len(subtasks)}.")
    return subtasks

def parse_context(text: str) -> str:
    match = re.search(r'FoamFile\s*\{.*?(?=```|$)', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).strip()
    
    print("Warning: Could not parse context; returning original text.")
    return text


def parse_file_name(subtask: str) -> str:
    match = re.search(r'openfoam\s+(.*?)\s+foamfile', subtask, re.IGNORECASE)
    return match.group(1).strip() if match else ""

def parse_folder_name(subtask: str) -> str:
    match = re.search(r'foamfile in\s+(.*?)\s+folder', subtask, re.IGNORECASE)
    return match.group(1).strip() if match else ""

def find_similar_file(description: str, tutorial: str) -> str:
    start_pos = tutorial.find(description)
    if start_pos == -1:
        return "None"
    end_marker = "input_file_end."
    end_pos = tutorial.find(end_marker, start_pos)
    if end_pos == -1:
        return "None"
    return tutorial[start_pos:end_pos + len(end_marker)]

def read_commands(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Commands file not found: {file_path}")
    with open(file_path, 'r') as f:
        # join non-empty lines with a comma
        return ", ".join(line.strip() for line in f if line.strip())

def find_input_file(case_dir: str, command: str) -> str:
    for root, _, files in os.walk(case_dir):
        for file in files:
            if command in file:
                return os.path.join(root, file)
    return ""

def retrieve_faiss(database_name: str, query: str, topk: int = 1) -> dict:
    """
    Retrieve a similar case from a FAISS database.
    """
    _ensure_faiss_loaded()

    if database_name not in FAISS_DB_CACHE:
        raise ValueError(f"Database '{database_name}' is not loaded.")

    # Tokenize the query
    query = tokenize(query)

    vectordb = FAISS_DB_CACHE[database_name]
    try:
        docs_and_scores = vectordb.similarity_search_with_score(query, k=topk)
        docs = [d for d, _ in docs_and_scores]
        scores = [s for _, s in docs_and_scores]
    except Exception:
        docs = vectordb.similarity_search(query, k=topk)
        scores = [None] * len(docs)

    if not docs:
        raise ValueError(f"No documents found for query: {query}")

    formatted_results = []
    for doc, score in zip(docs, scores):
        metadata = doc.metadata or {}

        if database_name == "openfoam_allrun_scripts":
            formatted_results.append({
                "index": doc.page_content,
                "full_content": metadata.get("full_content", "unknown"),
                "case_name": metadata.get("case_name", "unknown"),
                "case_domain": metadata.get("case_domain", "unknown"),
                "case_category": metadata.get("case_category", "unknown"),
                "case_solver": metadata.get("case_solver", "unknown"),
                "dir_structure": metadata.get("dir_structure", "unknown"),
                "allrun_script": metadata.get("allrun_script", "N/A"),
                "score": score,
            })
        elif database_name == "openfoam_command_help":
            formatted_results.append({
                "index": doc.page_content,
                "full_content": metadata.get("full_content", "unknown"),
                "command": metadata.get("command", "unknown"),
                "help_text": metadata.get("help_text", "unknown"),
                "score": score,
            })
        elif database_name == "openfoam_tutorials_structure":
            formatted_results.append({
                "index": doc.page_content,
                "full_content": metadata.get("full_content", "unknown"),
                "case_name": metadata.get("case_name", "unknown"),
                "case_domain": metadata.get("case_domain", "unknown"),
                "case_category": metadata.get("case_category", "unknown"),
                "case_solver": metadata.get("case_solver", "unknown"),
                "dir_structure": metadata.get("dir_structure", "unknown"),
                "score": score,
            })
        elif database_name == "openfoam_tutorials_details":
            formatted_results.append({
                "index": doc.page_content,
                "full_content": metadata.get("full_content", "unknown"),
                "case_name": metadata.get("case_name", "unknown"),
                "case_domain": metadata.get("case_domain", "unknown"),
                "case_category": metadata.get("case_category", "unknown"),
                "case_solver": metadata.get("case_solver", "unknown"),
                "dir_structure": metadata.get("dir_structure", "unknown"),
                "tutorials": metadata.get("tutorials", "N/A"),
                "score": score,
            })
        else:
            raise ValueError(f"Unknown database name: {database_name}")

    return formatted_results
        

def parse_directory_structure(data: str) -> dict:
    """
    Parses the directory structure string and returns a dictionary where:
      - Keys: directory names
      - Values: count of files in that directory.
    """
    directory_file_counts = {}

    # Find all <dir>...</dir> blocks in the input string.
    dir_blocks = re.findall(r'<dir>(.*?)</dir>', data, re.DOTALL)

    for block in dir_blocks:
        # Extract the directory name (everything after "directory name:" until the first period)
        dir_name_match = re.search(r'directory name:\s*(.*?)\.', block)
        # Extract the list of file names within square brackets
        files_match = re.search(r'File names in this directory:\s*\[(.*?)\]', block)
        
        if dir_name_match and files_match:
            dir_name = dir_name_match.group(1).strip()
            files_str = files_match.group(1)
            # Split the file names by comma, removing any surrounding whitespace
            file_list = [filename.strip() for filename in files_str.split(',')]
            directory_file_counts[dir_name] = len(file_list)

    return directory_file_counts
