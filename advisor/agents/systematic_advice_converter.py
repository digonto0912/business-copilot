from prompts import get_systematic_advice_converter_prompt
from schemas.systematic_advice_schema import BusinessStrategyBrief
from llm import llm


# ============================================================
# STRUCTURED OUTPUT
# ============================================================

structured_llm = llm.with_structured_output(
    BusinessStrategyBrief
)


# ============================================================
# AGENT
# ============================================================

systematic_advice_converter = (
    get_systematic_advice_converter_prompt
    | structured_llm
)