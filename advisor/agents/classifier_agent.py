from prompts import get_classifier_prompt
from llm import llm


classifier_agent = (
    get_classifier_prompt
    | llm
)