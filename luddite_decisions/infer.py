"""Use a Luddite Decisions model.

    from luddite_decisions.infer import DecisionModel
    model = DecisionModel("path/to/model")            # quant="nf4" to load in 4-bit
    model.decide({"enemy_in_sight": True, "ammo": 0}, "What should the guard do?",
              ["reload", "fire", "take cover"])
    -> {"pick": "reload", "probs": {"reload": 0.91, "fire": 0.02, "take cover": 0.07}}

One forward pass per question. decide_many() batches several questions.
"""

from __future__ import annotations

import sys
from pathlib import Path


from luddite_decisions.model import load, load_head, decisions_config, score  # noqa: E402


class DecisionModel:
    def __init__(self, path: str | Path, quant: str = "bf16"):
        self.tok, self.model = load(path, quant)
        self.model.eval()
        self.temperature = decisions_config(path).get("temperature", 1.0)
        self.head = load_head(path, self.model)

    def decide_many(self, items: list[dict]) -> list[dict]:
        """items: [{"state": ..., "question": str, "options": [str, ...]}]"""
        rows = [{"state": it.get("state", ""), "question": it["question"], "options": list(it["options"])} for it in items]
        out = []
        for r, p in zip(rows, score(self.model, self.tok, rows, temperature=self.temperature, head=self.head)):
            best = max(range(len(p)), key=p.__getitem__)
            out.append({"pick": r["options"][best], "probs": dict(zip(r["options"], (round(x, 4) for x in p)))})
        return out

    def decide(self, state, question: str, options: list[str]) -> dict:
        return self.decide_many([{"state": state, "question": question, "options": options}])[0]
