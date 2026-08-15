from prompts import get_problem_identifier_prompt
from schemas.problem_identifier_schema import (
    ProblemIdentifierResult,
)
from llm import llm


# ============================================================
# STRUCTURED OUTPUT
# ============================================================

structured_llm = llm.with_structured_output(
    ProblemIdentifierResult
)


# ============================================================
# AGENT
# ============================================================

problem_identifier_agent = (
    get_problem_identifier_prompt
    | structured_llm
)