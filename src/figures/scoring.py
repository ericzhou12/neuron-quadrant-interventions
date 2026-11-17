"""Accuracy scored on the last boxed answer anywhere in the output.

The standard score (`correct` in the result files) only reads the final answer
section, so a run that reaches the token cap while still thinking is wrong even
if it had already boxed the right answer. This score accepts the last
\\boxed{...} found in the thinking block or the answer; with no box anywhere it
falls back to the standard score.
"""
import re

_BOX = re.compile(r"\\boxed\{([^{}]*)\}")
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def last_boxed_correct(rec: dict) -> bool:
    boxes = _BOX.findall((rec.get("thinking") or "") + " " + (rec.get("answer_text") or ""))
    if not boxes:
        return bool(rec["correct"])
    nums = _NUM.findall(boxes[-1])
    if not nums:
        return False
    try:
        return abs(float(nums[-1].replace(",", "").rstrip(".")) - float(rec["gold"])) < 1e-6
    except ValueError:
        return False
