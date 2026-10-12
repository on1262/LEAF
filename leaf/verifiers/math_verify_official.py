"""math-verify route for the MATH500 benchmark verifier.

Thin wrapper around the ``math-verify`` package with the conservative
extractor of :mod:`leaf.verifiers.math_grader` as a fallback; benchmark-only.
"""
from math_verify import LatexExtractionConfig,parse,verify
from leaf.verifiers.math_grader import extract_boxed_or_final_answer,verify_math_symbolic


def verify_math(response,reference):
    prediction=extract_boxed_or_final_answer(response)
    if prediction is None:return False
    try:
        gold=parse(f'${reference}$',extraction_config=[LatexExtractionConfig()],fallback_mode='no_fallback',parsing_timeout=5)
        answer=parse(f'${prediction}$',extraction_config=[LatexExtractionConfig()],fallback_mode='no_fallback',parsing_timeout=5)
        if gold and answer:return bool(verify(gold,answer,timeout_seconds=5))
    except Exception:
        pass
    return verify_math_symbolic(response,reference)
