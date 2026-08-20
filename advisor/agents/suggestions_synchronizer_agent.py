import json

from langchain_core.output_parsers import StrOutputParser

from llm import llm_gemini_3_5_flash
from prompts import get_suggestions_synchronizer_prompt


class SuggestionsSynchronizerAgent:
    """Convert a completed reasoning tree into one user-facing strategy."""

    def __init__(self, model=None):
        self.model = model or llm_gemini_3_5_flash
        self.chain = get_suggestions_synchronizer_prompt | self.model | StrOutputParser()

    def synchronize(self, problem_tree):
        return self.chain.invoke(
            {
                "problem_tree": json.dumps(
                    problem_tree,
                    indent=2,
                    ensure_ascii=False,
                ),
            }
        )


suggestions_synchronizer_agent = SuggestionsSynchronizerAgent()
