"""Golden Q&A pairs for I-CARE MCP semantic eval.

Each entry defines a question, which tools are expected to be called,
and a programmatic check function that takes the LLM's final answer
(a string) and returns (passed: bool, reason: str).

The checks are deliberately lenient — we are testing whether the LLM
uses the tools correctly and produces a factually-grounded answer, not
whether it uses exact phrasing.
"""
from typing import Callable, List, Tuple

# ---- Known RT names (subset that must appear in Q1 answer) ----
KNOWN_RT_NAMES = [
    "SimilarityMatrix", "InterferenceMatrix", "EmbeddingForgettingEfficiency",
    "CountSignificantRelationship", "ImplicitAssociationTest",
]

# ---- Known people-task entity names (subset that must appear in Q2 answer) ----
KNOWN_PEOPLE_ENTITIES = [
    "George", "Tony Blair", "Colin Powell", "Vladimir Putin",
    "Tiger Woods", "Serena Williams",
]


def _check_q1(answer: str) -> Tuple[bool, str]:
    """Q1: LLM lists available RT names — at least 3 known names must appear."""
    found = [name for name in KNOWN_RT_NAMES if name.lower() in answer.lower()]
    if len(found) >= 3:
        return True, f"Found {len(found)}/5 expected RT names: {found}"
    return False, f"Only found {len(found)}/5 expected RT names: {found}"


def _check_q2(answer: str) -> Tuple[bool, str]:
    """Q2: LLM lists people-task entities — at least 4 known names must appear.

    Entity names may appear with spaces OR underscores (George W. Bush / George_W_Bush).
    """
    # Normalize: replace underscores with spaces for matching
    answer_normalized = answer.replace("_", " ").lower()
    found = [name for name in KNOWN_PEOPLE_ENTITIES if name.lower() in answer_normalized]
    if len(found) >= 4:
        return True, f"Found {len(found)}/6 expected entity names: {found}"
    return False, f"Only found {len(found)}/6 expected entity names: {found}"


def _check_q3(answer: str) -> Tuple[bool, str]:
    """Q3: LLM identifies the most similar entity to GWB — must name exactly one entity."""
    # The answer should contain at least one person's name that isn't GWB.
    # We just check it's not empty and doesn't say "I don't know".
    lowered = answer.lower()
    if len(answer.split()) < 5:
        return False, "Answer too short — LLM probably did not use the tool"
    if "don't know" in lowered or "cannot" in lowered or "error" in lowered:
        return False, "LLM expressed uncertainty or error"
    # Check it contains at least one proper noun (capitalized word sequence)
    import re
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer)
    proper_nouns = [p for p in proper_nouns if p not in ("The", "This", "Based", "According")]
    if proper_nouns:
        return True, f"LLM named entity: {proper_nouns[:3]}"
    return False, "No proper noun found in answer — LLM may not have read the matrix"


def _check_q4(answer: str) -> Tuple[bool, str]:
    """Q4: LLM reports a count of significantly affected entities — answer must contain a number."""
    import re
    numbers = re.findall(r'\b\d+\b', answer)
    if numbers:
        return True, f"Answer contains numbers: {numbers[:5]}"
    return False, "No number found in answer"


def _check_q5(answer: str) -> Tuple[bool, str]:
    """Q5: LLM describes interference matrix patterns — answer must be substantive (>40 words)."""
    word_count = len(answer.split())
    if word_count < 40:
        return False, f"Answer too short ({word_count} words) — LLM did not produce a real analysis"
    return True, f"Answer is substantive ({word_count} words)"


# ---- Golden Q&A registry ----
# Each entry: (id, question, expected_tools, check_fn)
GoldenQA = List[Tuple[str, str, List[str], Callable[[str], Tuple[bool, str]]]]

GOLDEN_QA: GoldenQA = [
    (
        "Q1",
        "What I-CARE analysis types (Result Templates) are available? List their names.",
        ["list_rts"],
        _check_q1,
    ),
    (
        "Q2",
        "What entities are in the 'people' task? Give me a sample of the names.",
        ["list_entities"],
        _check_q2,
    ),
    (
        "Q3",
        (
            "Using the jaccard similarity matrix for the people task, "
            "which entity is most similar to George W. Bush? "
            "Give me only the name of the most similar entity."
        ),
        ["compute_rt"],
        _check_q3,
    ),
    (
        "Q4",
        (
            "Use the interference matrix (clip_diff metric, distil method, people task) "
            "to tell me: how many entities have a clip_diff value below -0.05 "
            "when George W. Bush is unlearned? Count the rows where "
            "the George_W_Bush column is below -0.05."
        ),
        ["compute_rt"],
        _check_q4,
    ),
    (
        "Q5",
        (
            "Compute the interference matrix for the people task using "
            "clip_diff metric and the distil unlearning method. "
            "Describe in a few sentences what the overall pattern looks like: "
            "is interference widespread or concentrated? Are some emitters "
            "more damaging than others?"
        ),
        ["compute_rt"],
        _check_q5,
    ),
]
