from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# STRICT BASE MODEL
# ============================================================

class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


# ============================================================
# PROBLEM EXPLAINER RESULT
# ============================================================

class ProblemExplainerResult(StrictModel):
    """
    Determines whether a conditional critic creates a distinct
    child problem and, if so, formulates that problem for the
    next advisor flow.
    """

    classification: Literal[
        "NEW_PROBLEM",
        "NO_NEW_PROBLEM",
    ] = Field(
        description=(
            "NEW_PROBLEM when the critic identifies a distinct "
            "issue that should be solved separately. "
            "NO_NEW_PROBLEM when the critic is only a condition, "
            "caveat, or unresolved assumption that does not "
            "justify a separate problem."
        ),
    )

    problem: Optional[str] = Field(
        default=None,
        description=(
            "A concise problem statement to send to the advisor "
            "if classification is NEW_PROBLEM. It must be derived "
            "from the critic and verified context, not invented."
        ),
    )

    reason: str = Field(
        description=(
            "Brief explanation of why this is or is not a separate "
            "problem. Do not solve the problem."
        ),
    )