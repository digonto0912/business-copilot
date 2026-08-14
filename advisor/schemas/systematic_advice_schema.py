from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ============================================================
# STRICT BASE MODEL
# ============================================================

class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


# ============================================================
# BUSINESS SUMMARY
# ============================================================

class PriceRangeBDT(StrictModel):

    min: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Lower bound of the product price in BDT. "
            "Use null if not stated in the advisor reply."
        ),
    )

    max: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Upper bound of the product price in BDT. "
            "Use null if not stated in the advisor reply."
        ),
    )

    @model_validator(mode="after")
    def validate_range(self):
        if (
            self.min is not None
            and self.max is not None
            and self.min > self.max
        ):
            raise ValueError(
                "price_range_bdt.min must be <= price_range_bdt.max"
            )

        return self


class BusinessSummary(StrictModel):

    business_model: Optional[str] = Field(
        default=None,
        description=(
            "How the business operates. "
            "Use null if not stated."
        ),
    )

    product_category: Optional[str] = Field(
        default=None,
        description=(
            "What is being sold. "
            "Use null if not stated."
        ),
    )

    price_range_bdt: Optional[PriceRangeBDT] = Field(
        default=None,
        description=(
            "Product price range in BDT. "
            "Use null if not stated."
        ),
    )

    target_customer: Optional[str] = Field(
        default=None,
        description=(
            "Primary customer segment. "
            "Use null if not stated."
        ),
    )

    market: Optional[str] = Field(
        default=None,
        description=(
            "Geographic market. "
            "Use null if not stated."
        ),
    )

    fulfillment: Optional[str] = Field(
        default=None,
        description=(
            "Who handles sourcing, inventory, and delivery. "
            "Use null if not stated."
        ),
    )

    constraints: List[str] = Field(
        default_factory=list,
        description=(
            "Business constraints explicitly mentioned "
            "in the advisor reply."
        ),
    )


# ============================================================
# REAL PROBLEM
# ============================================================

class RealProblem(StrictModel):

    surface_problem: Optional[str] = Field(
        default=None,
        description=(
            "The problem as originally framed. "
            "Use null if unavailable."
        ),
    )

    actual_problem: Optional[str] = Field(
        default=None,
        description=(
            "The advisor's reframed underlying problem. "
            "Use null if not provided."
        ),
    )

    supporting_context: List[str] = Field(
        default_factory=list,
        description=(
            "Facts or evidence explicitly used by the "
            "advisor to support the reframing."
        ),
    )


# ============================================================
# STRATEGY REASONING
# ============================================================

class StrategyReasoning(StrictModel):

    core_constraint: Optional[str] = Field(
        default=None,
        description=(
            "Main constraint driving the strategy. "
            "Use null if not stated."
        ),
    )

    approach: Optional[str] = Field(
        default=None,
        description=(
            "Overall strategic approach stated by "
            "the advisor. Use null if not stated."
        ),
    )

    channels: List[str] = Field(
        default_factory=list,
        description=(
            "Channels or tactics explicitly mentioned "
            "by the advisor."
        ),
    )


# ============================================================
# ACTION PLAN
# ============================================================

class ActionPlanItem(StrictModel):

    priority: int = Field(
        ...,
        ge=1,
        description=(
            "Execution order. 1 is highest priority."
        ),
    )

    title: str = Field(
        ...,
        description="Short name of the action.",
    )

    action: str = Field(
        ...,
        description="What the business should do.",
    )

    execution: List[str] = Field(
        default_factory=list,
        description=(
            "Concrete execution steps explicitly "
            "provided by the advisor."
        ),
    )


# ============================================================
# RISKS
# ============================================================

class Risk(StrictModel):

    risk: str = Field(
        ...,
        description="Identified risk.",
    )

    impact: str = Field(
        ...,
        description="Impact if the risk occurs.",
    )

    mitigation: Optional[str] = Field(
        default=None,
        description=(
            "Mitigation provided by the advisor. "
            "Use null if none was given."
        ),
    )


# ============================================================
# ASSUMPTIONS / RISKS / TRADE-OFFS
# ============================================================

class AssumptionsRisksTradeoffs(StrictModel):

    assumptions: List[str] = Field(
        default_factory=list,
        description=(
            "Assumptions explicitly stated by the advisor."
        ),
    )

    risks: List[Risk] = Field(
        default_factory=list,
        description=(
            "Risks explicitly identified by the advisor."
        ),
    )

    tradeoffs: List[str] = Field(
        default_factory=list,
        description=(
            "Trade-offs explicitly described by the advisor."
        ),
    )


# ============================================================
# MODEL OUTPUT SCHEMA
# ============================================================
class BusinessStrategyBrief(StrictModel):
    title: str = Field(
        ...,
        min_length=1,
        description="Short descriptive title for the overall strategy.",
    )

    business_summary: BusinessSummary = Field(
        ...,
        description="Key business facts available in the advisor reply.",
    )

    real_problem: RealProblem = Field(
        ...,
        description="Stated problem and advisor's reframing.",
    )

    strategy_reasoning: StrategyReasoning = Field(
        ...,
        description="Constraint-driven reasoning behind the strategy.",
    )

    prioritized_action_plan: List[ActionPlanItem] = Field(
        default_factory=list,
        description="Ordered recommendations from the advisor.",
    )

    assumptions_risks_tradeoffs: AssumptionsRisksTradeoffs = Field(
        default_factory=AssumptionsRisksTradeoffs,
        description="Assumptions, risks, and trade-offs.",
    )
    """
    Structured representation of the advisor's final
    strategic recommendation.

    `document_type` is intentionally NOT generated by the LLM.
    It is added by the application after structured parsing.
    """

    title: str = Field(
        ...,
        min_length=1,
        description=(
            "Short descriptive title for the overall strategy."
        ),
    )

    business_summary: BusinessSummary = Field(
        ...,
        description=(
            "Key business facts available in the advisor reply."
        ),
    )

    real_problem: RealProblem = Field(
        ...,
        description=(
            "Stated problem and advisor's reframing."
        ),
    )

    strategy_reasoning: StrategyReasoning = Field(
        ...,
        description=(
            "Constraint-driven reasoning behind the strategy."
        ),
    )

    prioritized_action_plan: List[ActionPlanItem] = Field(
        default_factory=list,
        description=(
            "Ordered recommendations from the advisor."
        ),
    )

    assumptions_risks_tradeoffs: AssumptionsRisksTradeoffs = Field(
        default_factory=AssumptionsRisksTradeoffs,
        description=(
            "Assumptions, risks, and trade-offs "
            "associated with the recommendation."
        ),
    )