# agent_workflow.py

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import (
    StrOutputParser,
    PydanticOutputParser,
)

from agents import (
    advisor_agent,
    classifier_agent,
    verified_facts_agent,
    systematic_advice_converter,
)

from schemas.verified_facts_schema import VerifiedFacts


# ============================================================
# PYDANTIC PARSER
# ============================================================

verified_facts_parser = PydanticOutputParser(pydantic_object=VerifiedFacts)


# ============================================================
# CHAINS
# ============================================================

advisor_chain = advisor_agent | StrOutputParser()


classifier_chain = classifier_agent | StrOutputParser()


verified_facts_chain = verified_facts_agent | verified_facts_parser


# ============================================================
# SYSTEMATIC ADVICE CONVERTER
# ============================================================
#
# systematic_advice_converter already contains:
#
#     prompt
#        ↓
#     structured_llm
#
# where structured_llm uses:
#
#     BusinessStrategyBrief
#
# Therefore we do NOT add another parser here.
#
# ============================================================

systematic_advice_converter_chain = systematic_advice_converter


# ============================================================
# LLM DEBUG CALLBACK
# ============================================================


class LLMDebugHandler(BaseCallbackHandler):
    """
    Captures:

    1. The actual messages sent to the Chat Model
    2. The raw response returned by the Chat Model
    """

    def __init__(self):

        self.llm_prompt = None
        self.llm_output = None

    # --------------------------------------------------------
    # ACTUAL PROMPT SENT TO CHAT MODEL
    # --------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized,
        messages,
        **kwargs,
    ):

        try:
            prompt_messages = []

            for batch in messages:
                for message in batch:
                    prompt_messages.append(
                        {
                            "role": getattr(
                                message,
                                "type",
                                "unknown",
                            ),
                            "content": message.content,
                        }
                    )

            self.llm_prompt = prompt_messages

        except Exception as e:
            self.llm_prompt = {"error": str(e)}

    # --------------------------------------------------------
    # RAW LLM OUTPUT
    # --------------------------------------------------------

    def on_llm_end(
        self,
        response,
        **kwargs,
    ):

        try:
            generation = response.generations[0][0]

            if hasattr(generation, "message"):
                message = generation.message

                self.llm_output = {
                    "role": getattr(
                        message,
                        "type",
                        "ai",
                    ),
                    "content": message.content,
                }

            else:
                self.llm_output = {
                    "role": "ai",
                    "content": str(generation.text),
                }

        except Exception as e:
            self.llm_output = {"error": str(e)}


# ============================================================
# MAIN AGENT WORKFLOW
# ============================================================


def run_workflow(history):
    """
    Complete workflow:

        Full conversation
              ↓
           Advisor
              ↓
      latest advisor reply
              +
      latest 2 human messages
              ↓
       Verified Facts Agent
              ↓
        Pydantic validation
              ↓
          Classifier
              ↓
       NEEDS_INPUT?
          /       \\
        YES        NO
         ↓          ↓
       stop    Systematic Advice
                    Converter
                       ↓
             BusinessStrategyBrief
    """

    # ========================================================
    # 1. ADVISOR
    # ========================================================

    advisor_debug = LLMDebugHandler()

    try:
        advisor_response = advisor_chain.with_config(callbacks=[advisor_debug]).invoke(
            {"messages": history}
        )

        advisor_status = "SUCCESS"

    except Exception as e:
        advisor_response = str(e)

        advisor_status = "ERROR"

    # ========================================================
    # 2. FIND LAST TWO HUMAN MESSAGES
    # ========================================================

    human_messages = [content for role, content in history if role == "user"]

    last_two_human_messages = human_messages[-2:]

    if len(last_two_human_messages) == 0:
        human_message_1 = ""
        human_message_2 = ""

    elif len(last_two_human_messages) == 1:
        human_message_1 = ""
        human_message_2 = last_two_human_messages[0]

    else:
        human_message_1 = last_two_human_messages[-2]

        human_message_2 = last_two_human_messages[-1]

    # ========================================================
    # 3. VERIFIED FACTS AGENT
    # ========================================================

    verified_facts_debug = LLMDebugHandler()

    try:
        verified_facts = verified_facts_chain.with_config(
            callbacks=[verified_facts_debug]
        ).invoke(
            {
                "human_message_1": human_message_1,
                "human_message_2": human_message_2,
                "advisor_reply": advisor_response,
            }
        )

        verified_facts_status = "SUCCESS"

    except Exception as e:
        verified_facts = None

        verified_facts_status = "ERROR"

        verified_facts_debug.llm_output = {"error": str(e)}

    # ========================================================
    # 4. CLASSIFIER
    # ========================================================

    classifier_debug = LLMDebugHandler()

    try:
        classification = classifier_chain.with_config(
            callbacks=[classifier_debug]
        ).invoke({"response": advisor_response})

        classification = classification.strip().upper()

        classifier_status = "SUCCESS"

    except Exception as e:
        classification = str(e)

        classifier_status = "ERROR"

    # ========================================================
    # 5. DECISION
    # ========================================================

    needs_input = "NEEDS_INPUT" in classification

    # ========================================================
    # 6. SYSTEMATIC ADVICE CONVERTER
    # ========================================================
    #
    # ONLY run this when the classifier determines that
    # the advisor has enough information.
    #
    # The converter receives the final advisor response,
    # NOT the whole conversation.
    #
    # Its output is already a BusinessStrategyBrief
    # Pydantic object because systematic_advice_converter
    # uses with_structured_output().
    #
    # ========================================================

    systematic_advice = None
    systematic_advice_debug = None
    systematic_advice_status = None

    if not needs_input:
        systematic_advice_debug = LLMDebugHandler()

        try:
            systematic_advice = systematic_advice_converter_chain.with_config(
                callbacks=[systematic_advice_debug]
            ).invoke({"advisor_reply": advisor_response})

            systematic_advice_status = "SUCCESS"

        except Exception as e:
            systematic_advice = None

            systematic_advice_status = "ERROR"

            systematic_advice_debug.llm_output = {"error": str(e)}

    # ========================================================
    # 7. DEBUG LOGS
    # ========================================================

    logs = [
        {
            "agent": "Advisor LLM",
            "prompt": (advisor_debug.llm_prompt),
            "output": (advisor_debug.llm_output),
            "status": advisor_status,
        },
        {
            "agent": "Verified Facts LLM",
            "prompt": (verified_facts_debug.llm_prompt),
            "output": (verified_facts_debug.llm_output),
            "status": verified_facts_status,
        },
        {
            "agent": "Classifier LLM",
            "prompt": (classifier_debug.llm_prompt),
            "output": (classifier_debug.llm_output),
            "status": classifier_status,
        },
    ]

    # --------------------------------------------------------
    # Add converter log only when it actually ran
    # --------------------------------------------------------

    if systematic_advice_debug is not None:
        logs.append(
            {
                "agent": "Systematic Advice Converter LLM",
                "prompt": systematic_advice_debug.llm_prompt,
                "output": systematic_advice_debug.llm_output,
                "status": systematic_advice_status,
            }
        )

    # ========================================================
    # 8. RETURN WORKFLOW RESULT
    # ========================================================

    return {
        "response": advisor_response,
        "classification": classification,
        "needs_input": needs_input,
        "verified_facts": verified_facts,
        "systematic_advice": systematic_advice,
        "logs": logs,
    }
