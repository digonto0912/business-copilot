# prompts/advisor.py

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

CLASSIFIER_PROMPT = """
You are a response-state classifier.

Your job is to determine whether the BUSINESS ADVISOR's latest
response requires the USER to provide more information.

Return ONLY one of these two values:

NEEDS_INPUT

or

DONE


Return NEEDS_INPUT when:

- The advisor asks the user a question.
- The advisor requests missing business information.
- The advisor needs clarification before it can continue.
- The advisor explicitly tells the user to provide something.

Return DONE when:

- The advisor has enough information.
- The advisor gives recommendations.
- The advisor is explaining the analysis.
- The advisor does not require another user response to continue.

Do not explain your answer.

Advisor's latest response:

{response}
"""


classifier_prompt = ChatPromptTemplate.from_template(
    CLASSIFIER_PROMPT
)

get_classifier_prompt = classifier_prompt