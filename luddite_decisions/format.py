"""The Luddite Decisions prompt format, shared by training, evaluation and inference.

A decision is a state (text or a JSON-able value), a question and 2 to 10 options. The prompt
ends at "Answer:" and the model's next-token logits for " A", " B", ... are softmaxed over the
options. One forward pass, no generation.
"""

from __future__ import annotations

import json

FORMAT_VERSION = 1
LETTERS = "ABCDEFGHIJ"


def state_text(state) -> str:
    if state is None or state == "":
        return ""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def render(state, question: str, options: list[str]) -> str:
    if not 2 <= len(options) <= len(LETTERS):
        raise ValueError(f"2 to {len(LETTERS)} options, got {len(options)}")
    lines = []
    s = state_text(state)
    if s:
        lines.append(f"State: {s}")
    lines.append(f"Question: {question}")
    lines += [f"{LETTERS[i]}. {o}" for i, o in enumerate(options)]
    lines.append("Answer:")
    return "\n".join(lines)


def letter_ids(tok) -> list[int]:
    ids = []
    for letter in LETTERS:
        enc = tok.encode(" " + letter, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"' {letter}' is {len(enc)} tokens in this tokenizer")
        ids.append(enc[0])
    return ids
