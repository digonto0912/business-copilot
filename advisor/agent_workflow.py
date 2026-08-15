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
    CriticAgent,
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


systematic_advice_converter_chain = systematic_advice_converter


# ============================================================
# CRITIC
# ============================================================

critic_agent = CriticAgent()


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


def run_workflow(
    history,
    verified_facts_memory=None,
):
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
                       ↓
             one critic run per action
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
    # 6. SYSTEMATIC ADVICE
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
    # 7. BUILD VERIFIED CONTEXT FOR CRITIC
    # ========================================================
    #
    # Existing verified facts from previous turns are supplied
    # by main.py.
    #
    # The CURRENT verified-facts result is also added so the
    # critic has the latest information.
    #
    # ========================================================

    critic_verified_context = []

    if verified_facts_memory:
        critic_verified_context.extend(verified_facts_memory)

    if verified_facts is not None:
        critic_verified_context.append(verified_facts.model_dump())

    # ========================================================
    # 8. CRITIC
    # ========================================================

    critic_results = []

    critic_status = "SKIPPED"

    if not needs_input and systematic_advice is not None:
        critic_status = "SUCCESS"

        advisor_strategy = systematic_advice.model_dump()

        all_actions = advisor_strategy.get("prioritized_action_plan", [])

        # ----------------------------------------------------
        # ONE ACTION AT A TIME
        # ----------------------------------------------------
        #
        # The critic receives:
        #
        # verified_context
        # +
        # advisor strategy context
        # +
        # ONLY ONE target action
        #
        # It does NOT receive the other actions.
        #
        # ----------------------------------------------------

        for action_item in all_actions:
            target_strategy = {
                key: value
                for key, value in (advisor_strategy.items())
                if key != ("prioritized_action_plan")
            }

            target_strategy["prioritized_action_plan"] = [action_item]

            try:
                critique = critic_agent.critique_action(
                    action_item=action_item,
                    verified_context=(critic_verified_context),
                    advisor_strategy=(target_strategy),
                )

                critic_results.append(
                    {
                        "action": action_item,
                        "critique": (critique.model_dump()),
                        "status": "SUCCESS",
                    }
                )

            except Exception as e:
                critic_status = "ERROR"

                critic_results.append(
                    {
                        "action": action_item,
                        "critique": None,
                        "status": "ERROR",
                        "error": str(e),
                    }
                )

    # ========================================================
    # 9. DEBUG LOGS
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
    # SYSTEMATIC ADVICE LOG
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

    # --------------------------------------------------------
    # CRITIC LOG
    # --------------------------------------------------------
    #
    # CriticAgent currently owns its internal tool-loop, so
    # this log stores the structured result instead of claiming
    # to be the exact provider prompt.
    #
    # --------------------------------------------------------

    for index, critic_result in enumerate(
        critic_results,
        start=1,
    ):
        logs.append(
            {
                "agent": f"Reality-Check Critic #{index}",
                "prompt": {
                    "verified_context": critic_verified_context,
                    "advisor_strategy": {
                        key: value
                        for key, value in (systematic_advice.model_dump().items())
                        if key != ("prioritized_action_plan")
                    }
                    if systematic_advice is not None
                    else None,
                    "target_action": critic_result["action"],
                },
                "output": critic_result["critique"],
                "status": critic_result["status"],
            }
        )

    # ========================================================
    # 10. RETURN
    # ========================================================

    return {
        "response": advisor_response,
        "classification": classification,
        "needs_input": needs_input,
        "verified_facts": verified_facts,
        "systematic_advice": systematic_advice,
        "critic_results": critic_results,
        "logs": logs,
    }
