#!/usr/bin/env python3
"""P1-19 — enforce that the feature layer is made of pure functions.

Features must be pure functions of point-in-time data: no I/O, no state, no
reference to the current date. Purity is what makes lookahead detectable by
READING the code rather than by testing for it — and given that P1's bugs are
silent, anything that turns a silent failure into a loud one is worth having.

The specific thing being caught: a feature that calls `date.today()` or reads a
file directly has escaped the as-of discipline. It will compute correctly today
and wrongly in every backtest, and no test will fail.

Runs in CI. About thirty lines of AST walk, and it will save you a week.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Names that mean "look at the wall clock" or "read the disk yourself".
BANNED_ATTRS = {
    "now",
    "today",
    "utcnow",
    "fromtimestamp",
    "time",
    "read_parquet",
    "read_csv",
    "read_sql",
    "open",
    "connect",
}
BANNED_MODULES = {"duckdb", "psycopg", "sqlite3", "requests", "httpx", "urllib"}
BANNED_CALLS = {"open", "input", "print"}


class PurityVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.violations: list[str] = []

    def _flag(self, node: ast.AST, what: str, why: str) -> None:
        self.violations.append(f"{self.path}:{node.lineno}: {what} — {why}")

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            if a.name.split(".")[0] in BANNED_MODULES:
                self._flag(node, f"imports {a.name}", "features must not do I/O")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = (node.module or "").split(".")[0]
        if mod in BANNED_MODULES:
            self._flag(node, f"imports from {node.module}", "features must not do I/O")
        if (node.module or "").startswith("asetpay_core.store"):
            self._flag(
                node,
                "imports the store",
                "features RECEIVE data, they do not fetch it — otherwise the "
                "as-of discipline is bypassed",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in BANNED_ATTRS:
            self._flag(
                node,
                f".{f.attr}()",
                "reads the wall clock or the disk; a feature must be a pure "
                "function of the data it is given",
            )
        elif isinstance(f, ast.Name) and f.id in BANNED_CALLS:
            self._flag(node, f"{f.id}()", "side effect inside a feature")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self._flag(node, "global statement", "features must be stateless")
        self.generic_visit(node)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    features = root / "packages" / "core" / "src" / "asetpay_core" / "features"

    if not features.exists():
        print(f"no feature package yet at {features} — nothing to check (P1-18)")
        return 0

    violations: list[str] = []
    for py in sorted(features.rglob("*.py")):
        v = PurityVisitor(py.relative_to(root))
        v.visit(ast.parse(py.read_text(), filename=str(py)))
        violations += v.violations

    if violations:
        print("Feature purity violations:\n", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        print(
            "\nA feature that reads the clock or the disk has escaped the as-of\n"
            "discipline. It will compute correctly today and wrongly in every\n"
            "backtest, and no test will fail. Pass the data in instead.",
            file=sys.stderr,
        )
        return 1

    print(f"feature purity: OK ({len(list(features.rglob('*.py')))} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
