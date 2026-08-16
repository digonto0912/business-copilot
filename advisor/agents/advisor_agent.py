from prompts import get_advisor_prompt
from llm import llm_gemma_4_31b_it


advisor_agent = (
    get_advisor_prompt
    | llm_gemma_4_31b_it
)