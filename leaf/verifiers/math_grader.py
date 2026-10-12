"""Conservative benchmark math verifier (boxed / final-answer extraction and
normalised comparison). Benchmark-only; the training-data verifier routes are
separate.
"""

from __future__ import annotations

import re
from typing import Any

import sympy


_FINAL_PATTERN = re.compile(r"(?i)(?:final answer|answer)\s*(?:is|:|=)\s*([^\n]+)")


def _boxed_values(text: str) -> list[str]:
    values = []
    marker = "\\boxed{"
    start = 0
    while (position := text.find(marker, start)) >= 0:
        content_start = position + len(marker)
        depth = 1
        cursor = content_start
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            values.append(text[content_start : cursor - 1])
        start = content_start
    return values


def extract_boxed_or_final_answer(completion: str) -> str | None:
    boxed = _boxed_values(completion)
    if boxed:
        return boxed[-1].strip().rstrip(". ")
    matches = _FINAL_PATTERN.findall(completion)
    if matches:
        value = matches[-1].strip().rstrip(". ")
        if value.startswith("**") and "**" in value[2:]:
            value = value[2 : value.index("**", 2)]
        else:
            closing_bold = re.match(r"^(.+?)\*\*(?:\s|$)", value)
            if closing_bold:
                value = closing_bold.group(1)
        return value.strip().rstrip(". ")
    lines = [line.strip() for line in completion.splitlines() if line.strip()]
    return lines[-1].rstrip(". ") if lines else None


def _normalise(value: Any) -> str:
    text = str(value).strip()
    text = text.replace("**", "").replace("__", "")
    while "\\\\" in text:
        text = text.replace("\\\\", "\\")
    text = text.replace("\\$", "").replace("$", "").replace(",", "")
    text = re.sub(r"\\(?:text|mathrm)\{([^{}]+)\}", r"\1", text)
    text = text.replace("\\%", "%")
    if text.endswith("%"):
        return f"({text[:-1]})/100"
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\sqrt([A-Za-z0-9.]+)", r"sqrt(\1)", text)
    text = re.sub(r"^[A-Za-z]\s*=\s*", "", text)
    text = text.replace("^", "**")
    text = re.sub(r"\s+", "", text)
    # AIME answers are conventionally stored as three-digit strings (for
    # example ``025``), while models normally emit the numerically equivalent
    # integer (``25``).  SymPy rejects Python-style decimal integers with a
    # leading zero, so canonicalize a standalone integer before symbolic
    # parsing.  Do not rewrite digits embedded in algebraic expressions.
    integer = re.fullmatch(r"([+-]?)(\d+)", text)
    if integer:
        sign, digits = integer.groups()
        digits = digits.lstrip("0") or "0"
        sign = "" if digits == "0" else sign
        return sign + digits
    return text


def _strip_simple_numeric_unit(value: Any) -> str | None:
    """Return a leading numeric answer followed only by a plain-text unit."""
    text = str(value).strip().replace("**", "").replace("__", "")
    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[+-]?\d+(?:\.\d*)?)?)"
        r"\s+[A-Za-z][A-Za-z -]*[.!]?",
        text,
    )
    return match.group(1) if match else None


def symbolic_or_numeric_equal(prediction: Any, gold: Any, *, tolerance: float = 1e-6) -> bool:
    if prediction is None or gold is None:
        return False
    left, right = _normalise(prediction), _normalise(gold)
    if left.casefold() == right.casefold():
        return True
    # ``math_verify`` accepts a boxed final computation such as
    # ``\boxed{25 + 8 = 33}`` for the scalar answer ``033``.  Match that
    # behavior when the final equality has a standalone numeric right-hand
    # side, without treating arbitrary prose containing ``=`` as an answer.
    if "=" in left and "=" not in right:
        final_rhs = left.rsplit("=", 1)[-1]
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", final_rhs):
            try:
                difference = sympy.simplify(sympy.sympify(final_rhs) - sympy.sympify(right))
                if difference == 0 or abs(float(difference)) <= tolerance:
                    return True
            except Exception:
                pass
    try:
        difference = sympy.simplify(sympy.sympify(left) - sympy.sympify(right))
        if difference == 0:
            return True
        return abs(float(difference)) <= tolerance
    except Exception:  # SymPy can raise parser-specific exceptions on arbitrary model text.
        numeric_without_unit = _strip_simple_numeric_unit(prediction)
        if numeric_without_unit is None:
            return False
        try:
            difference = sympy.simplify(
                sympy.sympify(_normalise(numeric_without_unit)) - sympy.sympify(right)
            )
            return difference == 0 or abs(float(difference)) <= tolerance
        except Exception:
            return False


def verify_math_symbolic(completion: str, gold_answer: Any) -> bool:
    return symbolic_or_numeric_equal(extract_boxed_or_final_answer(completion), gold_answer)


def verify_math(completion: str, gold_answer: Any) -> bool:
    from leaf.verifiers.math_verify_official import verify_math as official
    return official(completion,gold_answer)
