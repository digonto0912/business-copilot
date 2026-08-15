from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# STRICT BASE MODEL
# ============================================================

class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


# ============================================================
# ASSUMPTION CLASSIFICATION
# ============================================================

AssumptionClassification = Literal[
    "VERIFIED",
    "CONTRADICTED",
    "CHECKABLE",
    "ASSUMPTION",
    "UNFALSIFIABLE",
]


# ============================================================
# STRUCTURAL FLAWS
# ============================================================

StructuralFlawType = Literal[
    "non_sequitur",
    "relevance_mismatch",
    "survivorship_bias",
    "resource_mismatch",
]


# ============================================================
# VERDICT
# ============================================================

Verdict = Literal[
    "PASS",
    "CONDITIONAL",
    "FAIL",
]


# ============================================================
# CLASSIFIED ASSUMPTION
# ============================================================

class ClassifiedAssumption(StrictModel):
    """
    One load-bearing assumption and its classification.
    """

    assumption: str = Field(
        ...,
        description=(
            "The load-bearing assumption stated plainly. "
            "It must be something that would break the mechanism "
            "if false, not a generic caveat."
        ),
    )

    classification: AssumptionClassification = Field(
        ...,
        description="Classification of the assumption.",
    )

    evidence: str = Field(
        ...,
        description=(
            "Evidence supporting the classification. "
            "VERIFIED: exact line from VERIFIED_CONTEXT. "
            "CONTRADICTED: exact conflicting line from VERIFIED_CONTEXT. "
            "CHECKABLE: precise factual question or search finding. "
            "ASSUMPTION: evidence that would confirm or kill the assumption. "
            "UNFALSIFIABLE: why it cannot be tested."
        ),
    )


# ============================================================
# STRUCTURAL FLAG
# ============================================================

class StructuralFlag(StrictModel):
    """
    One logical-structure problem.
    """

    flaw_type: StructuralFlawType = Field(
        ...,
        description="Type of logical structure flaw.",
    )

    explanation: str = Field(
        ...,
        description=(
            "Why this flaw applies specifically here. "
            "Name the bottleneck, resource, or constraint "
            "that the suggestion ignores or conflicts with."
        ),
    )


# ============================================================
# ACTION CRITIQUE
# ============================================================

class ActionCritique(StrictModel):
    """
    Complete reality-check critique for ONE action-plan item.
    """

    action_title: str = Field(
        ...,
        description=(
            "Title of the action item being evaluated, copied as given."
        ),
    )

    claim: str = Field(
        ...,
        description=(
            "Stage 1 CLAIM: the action being recommended, in one line."
        ),
    )

    mechanism: str = Field(
        ...,
        description=(
            "Stage 1 MECHANISM: why the author believes "
            "the action produces the intended outcome."
        ),
    )

    load_bearing_assumptions: List[ClassifiedAssumption] = Field(
        ...,
        min_length=1,
        max_length=5,
        description=(
            "Stage 2 and 3: maximum five load-bearing assumptions."
        ),
    )

    structural_flags: List[StructuralFlag] = Field(
        default_factory=list,
        description=(
            "Stage 4 logical structure problems. "
            "Empty when none are found."
        ),
    )

    settled_as_fact_violation: Optional[str] = Field(
        default=None,
        description=(
            "If a load-bearing assumption was presented as an established "
            "fact by the generator instead of being treated as uncertain, "
            "state that assumption here. Otherwise null."
        ),
    )

    verdict: Verdict = Field(
        ...,
        description="Stage 5 verdict.",
    )

    verdict_reason: str = Field(
        ...,
        description=(
            "2-3 sentences maximum. Name the specific assumption or "
            "structural flaw driving the verdict."
        ),
    )