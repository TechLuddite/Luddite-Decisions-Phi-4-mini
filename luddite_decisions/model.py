"""Loading and scoring shared by training, evaluation and inference."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from luddite_decisions.format import LETTERS, letter_ids, render

LORA_TARGETS = {
    "phi3": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    "bitnet": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def load(path: str | Path, quant: str = "bf16", dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw: dict = {"dtype": dtype, "device_map": "cuda"}
    if quant == "int8":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(path, **kw)
    return tok, model


class OptionHead(torch.nn.Module):
    """Scores each option from the hidden state at the end of its line and at "Answer:".

    For models that do not bind answers to letters (GPT-2). Still one forward pass."""

    def __init__(self, d: int, hidden: int = 512):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(3 * d, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 1))

    def forward(self, h_opt: torch.Tensor, h_last: torch.Tensor) -> torch.Tensor:
        h_last = h_last.expand_as(h_opt)
        return self.net(torch.cat([h_opt, h_last, h_opt * h_last], -1)).squeeze(-1)


def backbone(model):
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    return getattr(inner, "transformer", None) or inner.model


def option_ends(tok, text: str, options: list[str]) -> list[int]:
    """Token index (unpadded) of the last token of each option line."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    offs = enc["offset_mapping"]
    ends, pos = [], 0
    for i, o in enumerate(options):
        line = f"{LETTERS[i]}. {o}"
        pos = text.index(line, pos) + len(line)
        ends.append(max(j for j, (a, b) in enumerate(offs) if a < pos and b > a))
    return ends


def head_scores(model, head, tok, rows: list[dict], max_len: int) -> list[torch.Tensor]:
    texts = [render(r["state"], r["question"], r["options"]) for r in rows]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len,
              add_special_tokens=True).to(model.device)
    pos = enc["attention_mask"].long().cumsum(-1) - 1
    h = backbone(model)(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                        position_ids=pos.clamp(min=0)).last_hidden_state.float()
    width = enc["input_ids"].shape[1]
    out = []
    for i, (r, t) in enumerate(zip(rows, texts)):
        pad = width - int(enc["attention_mask"][i].sum())
        idx = torch.tensor([pad + j for j in option_ends(tok, t, r["options"])], device=h.device)
        out.append(head(h[i, idx], h[i, -1]))
    return out


def decisions_config(path: str | Path) -> dict:
    p = Path(path) / "decisions_config.json"
    return json.loads(p.read_text()) if p.exists() else {"temperature": 1.0}


def last_logits(model, enc) -> torch.Tensor:
    """Logits at the last position only (all prompts are left-padded)."""
    try:
        out = model(**enc, logits_to_keep=1)
    except TypeError:
        out = model(**enc)
    return out.logits[:, -1].float()


def option_logits(model, tok, ids, rows: list[dict], max_len: int) -> list[torch.Tensor]:
    texts = [render(r["state"], r["question"], r["options"]) for r in rows]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len,
              add_special_tokens=True).to(model.device)
    # Left padding: count positions from each prompt's first real token (GPT-2 uses absolute positions).
    pos = enc["attention_mask"].long().cumsum(-1) - 1
    enc["position_ids"] = pos.clamp(min=0)
    logits = last_logits(model, enc)
    return [logits[i, ids[: len(r["options"])]] for i, r in enumerate(rows)]


def load_head(path: str | Path, model) -> OptionHead | None:
    p = Path(path) / "decisions_head.safetensors"
    if not p.exists():
        return None
    from safetensors.torch import load_file
    w = load_file(p)
    head = OptionHead(w["net.0.weight"].shape[1] // 3, w["net.0.weight"].shape[0]).to(model.device)
    head.load_state_dict(w)
    return head.eval()


@torch.inference_mode()
def score(model, tok, rows: list[dict], batch: int = 16, max_len: int = 1024, temperature: float = 1.0,
          reverse: bool = False, head: OptionHead | None = None) -> list[list[float]]:
    """Probabilities per option for every row, in the row's own option order."""
    ids = letter_ids(tok)
    out = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        if reverse:
            chunk = [{**r, "options": r["options"][::-1]} for r in chunk]
        lgs = head_scores(model, head, tok, chunk, max_len) if head is not None else option_logits(model, tok, ids, chunk, max_len)
        for r, lg in zip(chunk, lgs):
            p = torch.softmax(lg / temperature, dim=0).tolist()
            out.append(p[::-1] if reverse else p)
    return out


def arch(model) -> str:
    return getattr(model.config, "model_type", "")


__all__ = ["LETTERS", "LORA_TARGETS", "OptionHead", "arch", "head_scores", "load", "load_head", "option_logits",
           "decisions_config", "score"]
