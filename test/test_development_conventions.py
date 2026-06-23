from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def python_files() -> list[Path]:
    ignored_parts = {".venv", "__pycache__", ".git"}
    return sorted(
        path
        for path in ROOT.rglob("*.py")
        if not ignored_parts.intersection(path.relative_to(ROOT).parts)
    )


class DevelopmentConventionTests(unittest.TestCase):
    def test_python_files_are_parseable(self) -> None:
        for path in python_files():
            with self.subTest(path=path.relative_to(ROOT)):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_python_files_have_clean_whitespace(self) -> None:
        for path in python_files():
            with self.subTest(path=path.relative_to(ROOT)):
                text = path.read_text(encoding="utf-8")
                self.assertTrue(text.endswith("\n"))
                self.assertNotIn("\r\n", text)
                for line_number, line in enumerate(text.splitlines(), start=1):
                    self.assertEqual(
                        line.rstrip(),
                        line,
                        f"{path.relative_to(ROOT)}:{line_number} has trailing whitespace",
                    )
                    self.assertFalse(
                        line.startswith("\t"),
                        f"{path.relative_to(ROOT)}:{line_number} uses tab indentation",
                    )


if __name__ == "__main__":
    unittest.main()
