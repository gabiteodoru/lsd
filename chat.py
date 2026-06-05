#!/usr/bin/env python3
"""
Token-windowed chat.

Commands:
  window <n>   set sliding window size in tokens (0 = disabled, uses KV cache)
  clear        wipe conversation history and start fresh
  quit         exit
  <anything else>  send as next user message
"""

import gc
import sys
import argparse
import torch
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

DEFAULT_MODEL = ".models/qwen2.5-7b-instruct"
DEFAULT_WINDOW = 0
DEFAULT_MAX_NEW = 0  # 0 = unlimited
TEMPERATURE = 0.7
SYSTEM_PROMPT = "You are a helpful assistant. Think through problems carefully and step by step."


@dataclass
class Token:
    id: int
    role: str  # "system" | "user" | "assistant" | "template"


class Chat:
    def __init__(self, model_name: str, window_size: int, log_path: Path):
        self.window_size = window_size  # 0 means no window (use cache)
        self.tokens: List[Token] = []
        self.messages: List[dict] = []
        self._past_key_values = None
        self._log = open(log_path, "a")
        self._log.write(f"\n=== session {datetime.now().isoformat()} ===\n")
        self._log.flush()
        self._load(model_name)
        self._add_message("system", SYSTEM_PROMPT)

    def _load(self, model_name: str):
        model_path = Path(model_name)
        if not model_path.exists():
            # Try resolving HF ID (e.g. "Qwen/Qwen2.5-7B-Instruct") to local path
            resolved = Path(".models") / model_name.split("/")[-1].lower()
            if resolved.exists():
                model_path = resolved
                model_name = str(resolved)
            else:
                print(f"ERROR: model '{model_name}' not found locally.", file=sys.stderr)
                print(f"Download it first with:", file=sys.stderr)
                print(f"  python3 download_model.py {model_name}", file=sys.stderr)
                sys.exit(1)
        vram_gb = sum(
            torch.cuda.get_device_properties(i).total_memory
            for i in range(torch.cuda.device_count())
        ) / 1e9 if torch.cuda.is_available() else 0
        shard_gb = sum(f.stat().st_size for f in model_path.glob("*.safetensors")) / 1e9
        shard_gb = shard_gb or sum(f.stat().st_size for f in model_path.glob("*.bin")) / 1e9
        import json
        cfg_path = model_path / "config.json"
        is_fp8 = False
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            is_fp8 = cfg.get("quantization_config", {}).get("quant_method") == "fp8"
        # fp8 loads at ~1x shard size; bf16 expands to ~2x
        vram_multiplier = 1.0 if is_fp8 else 2.0
        if shard_gb * 0.25 > vram_gb:
            print(f"ERROR: not enough VRAM for 4-bit ({vram_gb:.0f}GB available, need ~{shard_gb * 0.25:.0f}GB).", file=sys.stderr)
            sys.exit(1)
        use_4bit = shard_gb * vram_multiplier > vram_gb
        mode = "4-bit quant" if use_4bit else ("fp8/native" if is_fp8 else "bfloat16")
        print(f"Loading {model_name} — model: {shard_gb:.0f}GB, VRAM: {vram_gb:.0f}GB, mode: {mode} ...", flush=True)
        gc_thresholds = gc.get_threshold()
        gc.set_threshold(100, 5, 5)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if use_4bit:
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, quantization_config=bnb, device_map="auto",
                    low_cpu_mem_usage=True, trust_remote_code=True,
                )
            else:
                extra = {}
                if is_fp8:
                    # transformers 4.46 doesn't recognize DeepSeek's "fp8" quant_method,
                    # so strip it from the config and load weights in their native fp8 dtype
                    from transformers import AutoConfig
                    model_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                    model_cfg.quantization_config = None
                    extra = {"config": model_cfg, "torch_dtype": torch.float8_e4m3fn}
                else:
                    extra = {"torch_dtype": torch.bfloat16}
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, device_map="auto",
                    low_cpu_mem_usage=True, trust_remote_code=True, **extra,
                )
            self.model.eval()
        except Exception as e:
            print(f"ERROR loading model: {e}", file=sys.stderr)
            raise
        finally:
            gc.set_threshold(*gc_thresholds)
            gc.collect()
            torch.cuda.empty_cache()
        gen_config = self.model.generation_config
        stop = gen_config.eos_token_id if gen_config.eos_token_id is not None else self.tokenizer.eos_token_id
        self.stop_ids = set(stop if isinstance(stop, list) else [stop])
        print(f"Ready. stop_ids={self.stop_ids}\n")

    def _tokenize_messages(self, messages: List[dict], add_generation_prompt: bool = False) -> List[int]:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _add_message(self, role: str, content: str):
        before = self._tokenize_messages(self.messages) if self.messages else []
        self.messages.append({"role": role, "content": content})
        after = self._tokenize_messages(self.messages)
        self.tokens += [Token(id=i, role=role) for i in after[len(before):]]

    def _add_generation_prompt(self):
        before = self._tokenize_messages(self.messages)
        after = self._tokenize_messages(self.messages, add_generation_prompt=True)
        self.tokens += [Token(id=i, role="template") for i in after[len(before):]]

    def _log_turn(self, role: str, text: str):
        self._log.write(f"\n[{role}] {datetime.now().isoformat()}\n{text}\n")
        self._log.flush()

    def add_user(self, text: str):
        self._log_turn("user", text)
        self._add_message("user", text)
        self._add_generation_prompt()

    def _close_assistant_turn(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
        before = self._tokenize_messages(self.messages[:-1], add_generation_prompt=True)
        after = self._tokenize_messages(self.messages)
        content_ids = self.tokenizer.encode(content, add_special_tokens=False)
        closing_ids = after[len(before) + len(content_ids):]
        self.tokens += [Token(id=i, role="template") for i in closing_ids]

    def _context_ids(self) -> List[int]:
        ids = [t.id for t in self.tokens]
        return ids if self.window_size == 0 else ids[-self.window_size:]

    def generate(self, max_new: int = DEFAULT_MAX_NEW) -> tuple[int, int]:
        gen_start = len(self.tokens)
        generated_ids = []
        use_cache = self.window_size == 0
        print("\n[assistant]", flush=True)

        if use_cache and self._past_key_values is not None:
            # Feed only the new tokens since last generation
            new_ids = [t.id for t in self.tokens[self._cache_len:]]
            input_ids = torch.tensor([new_ids], device=self.model.device)
            past = self._past_key_values
        else:
            input_ids = torch.tensor([self._context_ids()], device=self.model.device)
            past = None

        for step in (range(max_new) if max_new > 0 else iter(int, 1)):
            if input_ids.shape[1] == 0:
                print("[WARNING] empty input, nothing to generate from.", file=sys.stderr)
                break

            try:
                with torch.no_grad():
                    out = self.model(input_ids=input_ids, past_key_values=past, use_cache=use_cache)
            except Exception as e:
                if past is not None and step == 0:
                    print(f"\n[WARNING] KV cache failed ({e}), retrying without cache ...", file=sys.stderr)
                    use_cache = False
                    past = None
                    input_ids = torch.tensor([self._context_ids()], device=self.model.device)
                    try:
                        with torch.no_grad():
                            out = self.model(input_ids=input_ids, past_key_values=None, use_cache=False)
                    except Exception as e2:
                        print(f"\n[ERROR at step {step}] {e2}", file=sys.stderr)
                        break
                else:
                    print(f"\n[ERROR at step {step}] {e}", file=sys.stderr)
                    break

            logits = out.logits[0, -1]
            past = out.past_key_values if use_cache else None

            probs = torch.softmax(logits / TEMPERATURE, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            if next_id in self.stop_ids:
                break

            self.tokens.append(Token(id=next_id, role="assistant"))
            generated_ids.append(next_id)
            text = self.tokenizer.decode([next_id], skip_special_tokens=False)
            print(text, end="", flush=True)

            if use_cache:
                input_ids = torch.tensor([[next_id]], device=self.model.device)
            else:
                input_ids = torch.tensor([self._context_ids()], device=self.model.device)

        print()
        gen_end = len(self.tokens) - 1
        content = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        self._log_turn("assistant", content)
        self._close_assistant_turn(content)

        if use_cache:
            self._past_key_values = past
            self._cache_len = len(self.tokens)

        return gen_start, gen_end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help="Sliding window size in tokens (0 = disabled, uses KV cache)")
    parser.add_argument("--max-new", type=int, default=DEFAULT_MAX_NEW,
                        help="Max tokens to generate (0 = unlimited, default: 0)")
    parser.add_argument("--log", default="conversation.log",
                        help="Path to append conversation log (default: conversation.log)")
    args = parser.parse_args()

    log_path = Path(args.log)
    print(f"Logging to {log_path.resolve()}")
    chat = Chat(args.model, args.window, log_path)
    status = "KV cache enabled" if chat.window_size == 0 else f"window={chat.window_size} tokens"
    print(f"{status}. Type 'help' for commands.\n")

    while True:
        try:
            user_input = input("[you] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        cmd = user_input.lower().split()

        if cmd[0] == "quit":
            print("Bye.")
            break
        elif cmd[0] == "help":
            print(__doc__)
        elif cmd[0] == "clear":
            chat.tokens.clear()
            chat.messages.clear()
            chat._past_key_values = None
            chat._cache_len = 0
            chat._add_message("system", SYSTEM_PROMPT)
            chat._log.write(f"\n[clear] {datetime.now().isoformat()}\n")
            chat._log.flush()
            print("Conversation cleared.")
        elif cmd[0] == "window" and len(cmd) == 2:
            n = int(cmd[1])
            chat.window_size = n
            chat._past_key_values = None  # invalidate cache when switching modes
            chat._log.write(f"\n[window] {datetime.now().isoformat()} {n}\n")
            chat._log.flush()
            print(f"Window {'disabled (KV cache on)' if n == 0 else f'set to {n} tokens (KV cache off)'}.")
        else:
            chat.add_user(user_input)
            start, end = chat.generate(args.max_new)
            print(f"\n[tokens {start}-{end}]")


if __name__ == "__main__":
    main()
