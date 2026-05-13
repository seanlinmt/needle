import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Silence TF/JAX noise before importing jax
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import jax
import jax.numpy as jnp
import numpy as np

from ..dataset.dataset import get_tokenizer, DEFAULT_MAX_ENC_LEN, DEFAULT_MAX_GEN_LEN
from ..model.architecture import SimpleAttentionNetwork, make_padding_mask
from ..model.run import (
    load_checkpoint,
    _get_decode_fn,
    _build_encoder_input,
    normalize_tools,
    restore_tool_names,
)

_DEFAULT_MAX_GEN_LEN = 512
_MAX_MAX_GEN_LEN = 4096
_MAX_JSON_BYTES = 1024 * 1024

_model = None
_params = None
_tokenizer = None
_lock = threading.Lock()


def _load_model(checkpoint_path):
    global _model, _params, _tokenizer
    _params, config = load_checkpoint(checkpoint_path)
    _model = SimpleAttentionNetwork(config)
    _tokenizer = get_tokenizer()


def generate_stream(
    model,
    params,
    tokenizer,
    query,
    tools="[]",
    max_gen_len=DEFAULT_MAX_GEN_LEN,
    max_enc_len=DEFAULT_MAX_ENC_LEN,
    normalize=True,
    constrained=True,
):
    """Generate tool-call output and yield individual token strings.

    Yields decoded token strings, then a final dict: {"__final__": normalized_text}.
    """
    name_map = {}
    if normalize:
        tools, name_map = normalize_tools(tools)

    enc_tokens = _build_encoder_input(tokenizer, query, tools, max_enc_len)
    enc_input = jnp.array([enc_tokens])

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    src_mask = make_padding_mask(enc_input, pad_id)
    encoder_out, enc_mask = model.apply(
        {"params": params}, enc_input, src_mask=src_mask, method="encode"
    )

    dec_buffer = jnp.full((1, max_gen_len), pad_id, dtype=jnp.int32)
    dec_buffer = dec_buffer.at[0, 0].set(eos_id)

    decode_fn = _get_decode_fn(model, max_gen_len)

    constrained_decoder = None
    if constrained:
        from needle.model.constrained import build_constrained_decoder

        constrained_decoder = build_constrained_decoder([tools], tokenizer)

    logits = decode_fn(params, dec_buffer, encoder_out, enc_mask)

    generated_tokens = []

    for i in range(0, max_gen_len - 1):
        next_logits = logits[0, i]

        if constrained_decoder and constrained_decoder.is_active(0):
            logits_np = np.array(next_logits)
            logits_np = constrained_decoder.constrain_logits(logits_np, 0)
            next_token = int(np.argmax(logits_np))
        else:
            next_token = int(jnp.argmax(next_logits))

        if constrained_decoder:
            constrained_decoder.update(0, next_token)

        if next_token == eos_id:
            break

        generated_tokens.append(next_token)
        dec_buffer = dec_buffer.at[0, i + 1].set(next_token)
        token_text = tokenizer.decode([next_token])
        if token_text != "<tool_call>":
            yield token_text

        logits = decode_fn(params, dec_buffer, encoder_out, enc_mask)

    result = tokenizer.decode(generated_tokens)
    if result.startswith("<tool_call>"):
        result = result[len("<tool_call>") :]
    if normalize and name_map:
        result = restore_tool_names(result, name_map)
    yield {"__final__": result}


def _read_request_body(handler, max_bytes):
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError:
        raise ValueError("invalid Content-Length")
    if length <= 0:
        raise ValueError("empty request body")
    if length > max_bytes:
        raise ValueError(f"request body too large (max {max_bytes} bytes)")
    return handler.rfile.read(length)


def _read_json_request(handler, max_bytes=_MAX_JSON_BYTES):
    try:
        raw = _read_request_body(handler, max_bytes)
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON") from exc
    if not isinstance(body, dict):
        raise ValueError("JSON body must be an object")
    return body


def _clamp_int(value, default, minimum, maximum, field_name):
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    return max(minimum, min(parsed, maximum))


def _convert_tools_openai_to_needle(tools):
    """Convert OpenAI-format tools to Needle's internal format."""
    if not tools:
        return "[]"
    result = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and "function" in tool:
            func = tool["function"]
            needle_tool = {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
            }
            params = func.get("parameters", {})
            if isinstance(params, dict):
                properties = params.get("properties", {})
                required = set(params.get("required", []))
                needle_params = {}
                for pname, prop in properties.items():
                    needle_prop = dict(prop)
                    needle_prop["required"] = pname in required
                    needle_params[pname] = needle_prop
                needle_tool["parameters"] = needle_params
            result.append(needle_tool)
        elif "name" in tool:
            result.append(tool)
    return json.dumps(result, separators=(",", ":"))


def _extract_query(messages):
    """Build a single query string from an OpenAI messages array."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages is required and must be a non-empty array")

    parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(content)
        elif role == "assistant":
            parts.append(f"Assistant: {content}")

    query = "\n".join(parts).strip()
    if not query:
        raise ValueError("query is empty")
    return query


def _parse_chat_request(body):
    query = _extract_query(body.get("messages"))

    tools = body.get("tools", [])
    if not isinstance(tools, list):
        tools = []
    needle_tools = _convert_tools_openai_to_needle(tools)

    tool_choice = body.get("tool_choice", "auto")
    if tool_choice == "none":
        needle_tools = "[]"

    max_tokens = _clamp_int(
        body.get("max_tokens", _DEFAULT_MAX_GEN_LEN),
        _DEFAULT_MAX_GEN_LEN,
        1,
        _MAX_MAX_GEN_LEN,
        "max_tokens",
    )
    stream = bool(body.get("stream", False))
    model_name = body.get("model", "needle")
    return query, needle_tools, max_tokens, stream, model_name


def _needle_output_to_tool_calls(output_text):
    """Parse Needle model output and convert to OpenAI tool_calls format."""
    try:
        calls = json.loads(output_text)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(calls, list):
        if isinstance(calls, dict) and "name" in calls:
            calls = [calls]
        else:
            return None

    tool_calls = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name", "")
        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, separators=(",", ":")),
                },
            }
        )
    return tool_calls if tool_calls else None


def _make_chat_completion_response(
    request_id, model_name, tool_calls, prompt_tokens, completion_tokens
):
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _make_sse_chunk(request_id, model_name, delta, finish_reason=None):
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


class _Handler(BaseHTTPRequestHandler):
    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/v1/models", "/models"):
            self._json_response(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "needle",
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "cactus-compute",
                        }
                    ],
                },
            )
            return
        self.send_error(404)

    def do_POST(self):
        path = self.path.split("?")[0]
        if path in ("/v1/chat/completions", "/chat/completions"):
            self._handle_chat_completions()
            return
        self.send_error(404)

    def _handle_chat_completions(self):
        try:
            body = _read_json_request(self)
            query, tools, max_tokens, stream, model_name = _parse_chat_request(body)
        except ValueError as exc:
            self._json_response(
                400,
                {"error": {"message": str(exc), "type": "invalid_request_error"}},
            )
            return

        request_id = f"chatcmpl-{uuid.uuid4().hex}"

        with _lock:
            if _model is None or _params is None or _tokenizer is None:
                self._json_response(
                    503,
                    {
                        "error": {
                            "message": "model is not loaded",
                            "type": "server_error",
                        }
                    },
                )
                return

            prompt_text = query
            if tools != "[]":
                prompt_text += " " + tools
            prompt_tokens = len(_tokenizer.encode(prompt_text))

            if stream:
                self._stream_response(
                    request_id, model_name, query, tools, max_tokens, prompt_tokens
                )
            else:
                self._sync_response(
                    request_id, model_name, query, tools, max_tokens, prompt_tokens
                )

    def _sync_response(
        self, request_id, model_name, query, tools, max_tokens, prompt_tokens
    ):
        try:
            tokens = []
            final_text = None
            for item in generate_stream(
                _model,
                _params,
                _tokenizer,
                query,
                tools=tools,
                max_gen_len=max_tokens,
                constrained=True,
            ):
                if isinstance(item, dict) and "__final__" in item:
                    final_text = item["__final__"]
                else:
                    tokens.append(item)

            if final_text is None:
                final_text = "".join(tokens)

            completion_tokens = len(tokens)
            tool_calls = _needle_output_to_tool_calls(final_text)

            if tool_calls is None or not tool_calls:
                response = _make_chat_completion_response(
                    request_id, model_name, [], prompt_tokens, completion_tokens
                )
                response["choices"][0]["message"]["content"] = final_text
                response["choices"][0]["message"]["tool_calls"] = None
            else:
                response = _make_chat_completion_response(
                    request_id, model_name, tool_calls, prompt_tokens, completion_tokens
                )

            self._json_response(200, response)
        except Exception as exc:
            self._json_response(
                500,
                {"error": {"message": str(exc), "type": "server_error"}},
            )

    def _stream_response(
        self, request_id, model_name, query, tools, max_tokens, prompt_tokens
    ):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self._set_cors_headers()
        self.end_headers()

        try:
            tokens = []
            final_text = None
            for item in generate_stream(
                _model,
                _params,
                _tokenizer,
                query,
                tools=tools,
                max_gen_len=max_tokens,
                constrained=True,
            ):
                if isinstance(item, dict) and "__final__" in item:
                    final_text = item["__final__"]
                else:
                    tokens.append(item)
                    chunk = _make_sse_chunk(request_id, model_name, {"content": item})
                    self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()

            finish_chunk = _make_sse_chunk(
                request_id, model_name, {}, finish_reason="stop"
            )
            self.wfile.write(finish_chunk.encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception:
            # Send finish chunk on error so clients don't hang
            finish_chunk = _make_sse_chunk(
                request_id, model_name, {}, finish_reason="stop"
            )
            self.wfile.write(finish_chunk.encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    def _json_response(self, code, data):
        payload = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


def _resolve_checkpoint(checkpoint_arg):
    from huggingface_hub import hf_hub_download

    local_dir = "checkpoints"
    os.makedirs(local_dir, exist_ok=True)
    if checkpoint_arg and os.path.exists(checkpoint_arg):
        return checkpoint_arg
    filename = os.path.basename(checkpoint_arg) if checkpoint_arg else "needle.pkl"
    repo = "Cactus-Compute/needle"
    print(f"Downloading {filename} from {repo}...", file=sys.stderr)
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        repo_type="model",
        local_dir=local_dir,
        force_download=True,
    )
    print(f"Downloaded to {path}", file=sys.stderr)
    return path


def main(args):
    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}", file=sys.stderr)
    _load_model(checkpoint_path)

    param_count = sum(x.size for x in jax.tree.leaves(_params))
    print(f"Model parameters: {param_count:,}", file=sys.stderr)

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    server.request_queue_size = 16
    print(
        f"Needle OpenAI API: http://{args.host}:{args.port}/v1/chat/completions",
        file=sys.stderr,
    )
    server.serve_forever()


def parse_args():
    parser = argparse.ArgumentParser(description="Needle OpenAI-compatible API server")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
