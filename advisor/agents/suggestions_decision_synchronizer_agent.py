import json

from langchain_core.output_parsers import StrOutputParser

from llm import llm_gemini_3_5_flash
from prompts import get_suggestions_decision_synchronizer_prompt


class SuggestionsDecisionSynchronizerAgent:
    """Compress a fully evaluated strategy into final per-action decisions."""

    def __init__(self, model=None):
        self.model = model or llm_gemini_3_5_flash
        self.chain = get_suggestions_decision_synchronizer_prompt | self.model | StrOutputParser()

    def synchronize(self, problem_tree, first_sync_response):
        return self.chain.invoke(
            {
                "problem_tree": json.dumps(problem_tree, indent=2, ensure_ascii=False),
                "first_sync_response": first_sync_response,
            }
        )


suggestions_decision_synchronizer_agent = SuggestionsDecisionSynchronizerAgent()
