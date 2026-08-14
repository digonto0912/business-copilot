from langchain_core.output_parsers import PydanticOutputParser

from prompts import get_verified_facts_prompt
from schemas.verified_facts_schema import VerifiedFacts
from llm import llm


# ============================================================
# PYDANTIC OUTPUT PARSER
# ============================================================

verified_facts_parser = PydanticOutputParser(
    pydantic_object=VerifiedFacts
)


# ============================================================
# PROMPT
# ============================================================

verified_facts_prompt = get_verified_facts_prompt.partial(
    format_instructions=(
        verified_facts_parser
        .get_format_instructions()
    )
)


# ============================================================
# AGENT
# ============================================================

verified_facts_agent = (
    verified_facts_prompt
    | llm
)