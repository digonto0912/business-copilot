from prompts import get_advisor_prompt
from llm import llm


advisor_agent = (
    get_advisor_prompt
    | llm
)