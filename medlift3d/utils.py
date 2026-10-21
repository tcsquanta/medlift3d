"""Small shared helpers: devices, seeding, timing, PASS/FAIL verification reports."""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from typing import Any, List, Optional

import numpy as np
import torch


def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def timer(name: str = "", store: Optional[dict] = None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    if store is not None:
        store[name] = dt
    else:
        print(f"[time] {name}: {dt:.2f}s")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class Checker:
    """Collects verification checks and prints PASS/FAIL lines.

    Every script ends with ``checker.finish(path)`` which prints a summary,
    writes ``<path>/verification.md`` + ``verification.json`` and returns
    True only if every check passed.
    """

    def __init__(self, title: str):
        self.title = title
        self.rows: List[dict] = []
        print("=" * 78)
        print(f"VERIFICATION: {title}")
        print("=" * 78)

    def check(self, name: str, ok: bool, value: Any = "", expect: str = "") -> bool:
        ok = bool(ok)
        self.rows.append({"check": name, "ok": ok, "value": _fmt(value), "expect": expect})
        tag = "PASS" if ok else "FAIL"
        extra = f"  (expected {expect})" if expect else ""
        print(f"[{tag}] {name}: {_fmt(value)}{extra}")
        return ok

    def info(self, name: str, value: Any = "") -> None:
        self.rows.append({"check": name, "ok": None, "value": _fmt(value), "expect": ""})
        print(f"[INFO] {name}: {_fmt(value)}")

    @property
    def all_ok(self) -> bool:
        return all(r["ok"] for r in self.rows if r["ok"] is not None)

    def finish(self, out_dir: Optional[str] = None) -> bool:
        n_pass = sum(1 for r in self.rows if r["ok"] is True)
        n_fail = sum(1 for r in self.rows if r["ok"] is False)
        print("-" * 78)
        print(f"{self.title}: {n_pass} passed, {n_fail} failed -> {'ALL PASS' if n_fail == 0 else 'SOME CHECKS FAILED'}")
        print("-" * 78)
        if out_dir:
            ensure_dir(out_dir)
            with open(os.path.join(out_dir, "verification.json"), "w") as f:
                json.dump({"title": self.title, "rows": self.rows, "all_ok": n_fail == 0}, f, indent=2)
            with open(os.path.join(out_dir, "verification.md"), "w") as f:
                f.write(f"# {self.title}\n\n| status | check | value | expected |\n|---|---|---|---|\n")
                for r in self.rows:
                    st = "INFO" if r["ok"] is None else ("PASS" if r["ok"] else "FAIL")
                    f.write(f"| {st} | {r['check']} | {r['value']} | {r['expect']} |\n")
        return n_fail == 0


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)) and v and all(isinstance(x, float) for x in v):
        return "[" + ", ".join(f"{x:.4g}" for x in v) + "]"
    return str(v)
