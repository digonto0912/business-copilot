from langchain_core.prompts import ChatPromptTemplate


SYSTEMATIC_ADVICE_CONVERTER_PROMPT = """
Convert the advisor reply into the BusinessStrategyBrief schema.

Extract only what is stated or clearly implied in the advisor reply.
Never invent facts.
"""


systematic_advice_converter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEMATIC_ADVICE_CONVERTER_PROMPT,
        ),
        (
            "human",
            "{advisor_reply}",
        ),
    ]
)


get_systematic_advice_converter_prompt = (
    systematic_advice_converter_prompt
)