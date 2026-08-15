import json

from prompts import get_problem_explainer_prompt
from schemas.problem_explainer_schema import (
    ProblemExplainerResult,
)
from llm import llm


# ============================================================
# STRUCTURED OUTPUT
# ============================================================

structured_llm = llm.with_structured_output(
    ProblemExplainerResult
)


# ============================================================
# AGENT
# ============================================================

problem_explainer_agent = (
    get_problem_explainer_prompt
    | structured_llm
)