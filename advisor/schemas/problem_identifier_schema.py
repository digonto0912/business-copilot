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
# PROBLEM IDENTIFIER RESULT
# ============================================================

class ProblemIdentifierResult(StrictModel):
    """
    Determines whether the latest user input starts a new problem
    or is simply part of the ongoing conversation / Q&A.
    """

    classification: Literal[
        "PROBLEM",
        "Q&A",
    ] = Field(
        description=(
            "PROBLEM when the user input introduces a business "
            "problem that should become a problem-stack item. "
            "Q&A when the input is an answer, clarification, "
            "additional context, or normal continuation of the "
            "current problem."
        ),
    )

    problem: Optional[str] = Field(
        default=None,
        description=(
            "The problem stated by the user, preserving the user's "
            "meaning. Required when classification is PROBLEM; "
            "null when classification is Q&A."
        ),
    )