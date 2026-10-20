"""Deterministic decode and precision-validation request workloads.

This module deliberately has no model, CUDA, vLLM, or network dependency.  A
caller supplies the tokenizer used by the serving model and receives plain
Python data suitable for JSON serialization.  Prompts are ordinary completion
text; no guessed chat-template or special-token IDs are inserted.
"""

from __future__ import annotations

from typing import Callable, Sequence


Tokenizer = Callable[[str], Sequence[int]]


_TOPICS = (
    "code",
    "math",
    "json",
    "logic",
    "english_retrieval",
    "chinese_retrieval",
    "science",
    "tabular",
    "planning",
    "spelling",
    "security",
    "algorithms",
    "history",
    "geography",
    "data_cleaning",
    "instruction_following",
)


def _task(topic: str, serial: int) -> tuple[str, str, str]:
    """Return varied reference text, a short final task, and a simple answer."""
    n = serial
    if topic == "code":
        return (
            f"Python record {n}: values=[{n % 7}, {(n + 3) % 11}, {(n + 5) % 13}]. "
            "The function should return the sum of the distinct values.",
            "What is the output for this record?",
            str(len({n % 7, (n + 3) % 11, (n + 5) % 13}) and sum({n % 7, (n + 3) % 11, (n + 5) % 13})),
        )
    if topic == "math":
        a, b = 17 + n % 31, 4 + n % 9
        return (f"Arithmetic card {n}: compute ({a} * {b}) + {n % 6}.", "Give the integer result.", str(a * b + n % 6))
    if topic == "json":
        value = (n * 7) % 97
        return (f'JSON card {n}: {{"id": {n}, "items": ["red", "blue"], "count": {value}}}.', "Return only the count field.", str(value))
    if topic == "logic":
        answer = "true"
        return (f"Logic card {n}: every cedar is a tree; item {n} is a cedar. Is item {n} a tree?", "Answer true or false.", answer)
    if topic == "english_retrieval":
        word = ("orbit", "harbor", "meadow", "copper")[n % 4]
        return (f"English note {n}: the marker word in this note is {word}; nearby distractor={n + 19}.", "Copy the marker word.", word)
    if topic == "chinese_retrieval":
        word = ("春天", "河流", "月亮", "森林")[n % 4]
        return (f"中文记录 {n}：关键词是“{word}”，编号为{n + 31}。", "只回答关键词。", word)
    if topic == "science":
        element = ("water", "carbon", "oxygen", "iron")[n % 4]
        return (f"Science note {n}: the listed sample is {element}; its label is S{n:04d}.", "Return the sample name.", element)
    if topic == "tabular":
        score = (n * 13 + 5) % 100
        return (f"Table row {n}: region=R{n % 5}, units={n + 8}, score={score}.", "Return score as a number.", str(score))
    if topic == "planning":
        day = ("Monday", "Tuesday", "Wednesday", "Thursday")[n % 4]
        return (f"Schedule {n}: the inspection is on {day}; room={100 + n % 17}.", "Return the inspection day.", day)
    if topic == "spelling":
        pair = (("separate", "seperate"), ("necessary", "neccessary"), ("calendar", "calender"))[n % 3]
        return (f"Dictionary card {n}: choose the correctly spelled form from {pair[0]} and {pair[1]}.", "Return the correct spelling.", pair[0])
    if topic == "security":
        return (f"Security label {n}: a password reset token is an opaque value; example id=T{n:05d}.", "Return the safe action: rotate.", "rotate")
    if topic == "algorithms":
        value = (n % 23) + 2
        return (f"Algorithm card {n}: binary search on a sorted list checks {value} candidate positions.", "Return the candidate count.", str(value))
    if topic == "history":
        year = 1900 + n % 100
        return (f"History card {n}: the fictional archive entry has year={year} and shelf=H{n % 9}.", "Return the year.", str(year))
    if topic == "geography":
        place = ("north", "south", "east", "west")[n % 4]
        return (f"Map note {n}: the marked trail continues toward {place}; map tile M{n:04d}.", "Return the direction.", place)
    if topic == "data_cleaning":
        return (f"Cleaning row {n}: raw value='  item-{n}  ', trim surrounding spaces and retain its hyphen.", "Return the cleaned value.", f"item-{n}")
    return (f"Instruction card {n}: preserve the label L{n:04d} and ignore the distractor {n + 1}.", "Return the preserved label.", f"L{n:04d}")


def _ids(tokenize: Tokenizer, text: str) -> list[int]:
    values = list(tokenize(text))
    if any(not isinstance(value, int) for value in values):
        raise TypeError("tokenize(text) must return a sequence of integer token IDs")
    return values


def _exact_prompt(tokenize: Tokenizer, target: int, prefix: str, body: str, suffix: str) -> list[int]:
    """Compose tokenizer-produced IDs, preserving the complete final task.

    The caller must tokenize without automatically inserting special tokens.
    Only repeated context is shortened to hit the exact boundary length.
    """
    if target <= 0:
        raise ValueError("target prompt length must be positive")
    prefix_ids = _ids(tokenize, prefix + "\n")
    suffix_ids = _ids(tokenize, "\n" + suffix)
    body_ids = _ids(tokenize, body + "\n")
    if len(prefix_ids) + len(suffix_ids) > target:
        raise ValueError(f"cannot fit intact final task into target {target}")
    remaining = target - len(prefix_ids) - len(suffix_ids)
    if not body_ids:
        raise ValueError('Tokenizer produced empty context')
    middle = (body_ids * (remaining // len(body_ids) + 1))[:remaining]
    return prefix_ids + middle + suffix_ids


def _request(tokenize: Tokenizer, *, key: str, target: int, topic: str, serial: int) -> dict:
    body, question, answer = _task(topic, serial)
    prefix = f"Record {serial}:"
    suffix = f"{body}\n{question}\nAnswer:"
    return {
        "key": key,
        "prompt_token_ids": _exact_prompt(tokenize, target, prefix, body, suffix),
        "topic": topic,
        "target_prompt_tokens": target,
        "expected_answer": answer,
    }


def build_workload(tokenize: Tokenizer) -> dict:
    """Return deterministic single-request and fixed-batch decode cases.

    ``tokenize`` is called with plain strings and must return integer IDs.  The
    result contains no tensors or non-JSON values and can be dumped directly
    with :func:`json.dump`.  Expected answers are tiny smoke-test gold labels,
    not an academic quality score.
    """
    cases: list[dict] = []
    serial = 0
    for target in (64, 127, 128, 129, 511, 512, 1023, 1024, 2048, 3072):
        topic = 'math' if target == 64 else _TOPICS[serial % len(_TOPICS)]
        key = f"b1_p{target}_r0"
        cases.append({
            "label": f"b1_prompt{target}_out160",
            "batch": 1,
            "output_tokens": 160,
            "requests": [_request(tokenize, key=key, target=target, topic=topic, serial=serial)],
        })
        serial += 1
    for batch in (2, 4, 8, 16, 32, 64, 128, 256):
        requests = []
        for row in range(batch):
            topic = _TOPICS[(serial + row) % len(_TOPICS)]
            requests.append(_request(
                tokenize,
                key=f"b{batch}_p2048_r{row:03d}",
                target=2048,
                topic=topic,
                serial=serial + row,
            ))
        serial += batch
        cases.append({"label": f"b{batch}_prompt2048_out32", "batch": batch, "output_tokens": 32, "requests": requests})
    return {"cases": cases}
