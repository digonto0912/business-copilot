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

    1. Actual messages sent to the Chat Model
    2. Raw response returned by the Chat Model
    """

    def __init__(self):

        self.llm_prompt = None
        self.llm_output = None

    # --------------------------------------------------------
    # ACTUAL PROMPT
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
    # RAW OUTPUT
    # --------------------------------------------------------

    def on_llm_end(
        self,
        response,
        **kwargs,
    ):

        try:
            generation = response.generations[0][0]

            if hasattr(
                generation,
                "message",
            ):
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
# CORE PIPELINE
# ============================================================


def run_core_workflow(
    history,
    verified_facts_memory=None,
    runtime=1,
    auto_runtime=0,
    current_problem=None,
    current_problem_id=None,
    critic_id_counter=0,
    active_critics=None,
):
    """
    Runs ONE complete core agent pipeline.

    This function does NOT:

    - create problems
    - push/pop problems
    - call Problem Identifier
    - call Problem Explainer
    - increment auto_runtime
    - recursively restart itself

    It only runs:

        Advisor
          ↓
        Verified Facts
          ↓
        Classifier
          ↓
        Systematic Advice Converter
          ↓
        ALL Critics

    The recursion controller handles what happens after this
    function returns.
    """

    # ========================================================
    # NORMALIZE MEMORY
    # ========================================================

    if verified_facts_memory is None:
        verified_facts_memory = []

    if active_critics is None:
        active_critics = []

    # ========================================================
    # RESULT CONTAINERS
    # ========================================================

    logs = []

    critic_results = []

    critic_records = []

    conditional_critics = []

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

        advisor_debug.llm_output = {"error": str(e)}

    logs.append(
        {
            "agent": "Advisor LLM",
            "runtime": runtime,
            "auto_runtime": auto_runtime,
            "problem_id": current_problem_id,
            "prompt": advisor_debug.llm_prompt,
            "output": advisor_debug.llm_output,
            "status": advisor_status,
        }
    )

    # ========================================================
    # 2. LAST TWO HUMAN MESSAGES
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
    # 3. VERIFIED FACTS
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

    logs.append(
        {
            "agent": "Verified Facts LLM",
            "runtime": runtime,
            "auto_runtime": auto_runtime,
            "problem_id": current_problem_id,
            "prompt": verified_facts_debug.llm_prompt,
            "output": verified_facts_debug.llm_output,
            "status": verified_facts_status,
        }
    )

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

        classifier_debug.llm_output = {"error": str(e)}

    logs.append(
        {
            "agent": "Classifier LLM",
            "runtime": runtime,
            "auto_runtime": auto_runtime,
            "problem_id": current_problem_id,
            "prompt": classifier_debug.llm_prompt,
            "output": classifier_debug.llm_output,
            "status": classifier_status,
        }
    )

    # ========================================================
    # 5. DECISION
    # ========================================================

    needs_input = "NEEDS_INPUT" in classification

    # ========================================================
    # STOP HERE IF MORE USER INPUT IS REQUIRED
    # ========================================================

    if needs_input:
        return {
            "response": advisor_response,
            "classification": classification,
            "needs_input": True,
            "verified_facts": verified_facts,
            "systematic_advice": None,
            "critic_results": [],
            "critic_records": [],
            "conditional_critics": [],
            "active_critics": active_critics,
            "critic_id_counter": critic_id_counter,
            "logs": logs,
        }

    # ========================================================
    # 6. SYSTEMATIC ADVICE CONVERTER
    # ========================================================

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

    logs.append(
        {
            "agent": "Systematic Advice Converter LLM",
            "runtime": runtime,
            "auto_runtime": auto_runtime,
            "problem_id": current_problem_id,
            "prompt": systematic_advice_debug.llm_prompt,
            "output": systematic_advice_debug.llm_output,
            "status": systematic_advice_status,
        }
    )

    # ========================================================
    # STOP IF CONVERTER FAILED
    # ========================================================

    if systematic_advice is None:
        return {
            "response": advisor_response,
            "classification": classification,
            "needs_input": False,
            "verified_facts": verified_facts,
            "systematic_advice": None,
            "critic_results": [],
            "critic_records": [],
            "conditional_critics": [],
            "active_critics": active_critics,
            "critic_id_counter": critic_id_counter,
            "logs": logs,
        }

    # ========================================================
    # 7. VERIFIED FACT MEMORY FOR CRITICS
    # ========================================================

    critic_verified_context = []

    if verified_facts_memory:
        critic_verified_context.extend(verified_facts_memory)

    if verified_facts is not None:
        critic_verified_context.append(verified_facts.model_dump())

    # ========================================================
    # 8. GET ADVISOR STRATEGY
    # ========================================================

    advisor_strategy = systematic_advice.model_dump()

    all_actions = advisor_strategy.get(
        "prioritized_action_plan",
        [],
    )

    # ========================================================
    # 9. RUN ALL CRITICS
    # ========================================================
    #
    # IMPORTANT:
    #
    # Every action gets its own critic.
    #
    # Conditional critic does NOT interrupt the other critics.
    #
    # Example:
    #
    # Critic 1 → FAIL
    # Critic 2 → CONDITIONAL
    # Critic 3 → PASS
    #
    # All three finish first.
    #
    # Only AFTER all critics finish does this function return
    # the conditional_critics queue to the recursion controller.
    #
    # ========================================================

    for action_item in all_actions:
        # ----------------------------------------------------
        # GLOBAL CRITIC ID
        # ----------------------------------------------------

        critic_id_counter += 1

        critic_id = critic_id_counter

        # ----------------------------------------------------
        # ACTIVE CRITIC RECORD
        # ----------------------------------------------------

        critic_record = {
            "critic_id": critic_id,
            "problem_id": current_problem_id,
            "runtime": runtime,
            "auto_runtime": auto_runtime,
        }

        active_critics.append(critic_record)

        # ----------------------------------------------------
        # TARGET STRATEGY
        # ----------------------------------------------------

        target_strategy = {
            key: value
            for key, value in (advisor_strategy.items())
            if key != ("prioritized_action_plan")
        }

        # IMPORTANT:
        #
        # Only ONE action is exposed to this critic.

        target_strategy["prioritized_action_plan"] = [action_item]

        # ----------------------------------------------------
        # RUN CRITIC
        # ----------------------------------------------------

        try:
            critique = critic_agent.critique_action(
                action_item=action_item,
                verified_context=critic_verified_context,
                advisor_strategy=target_strategy,
            )

            critique_data = critique.model_dump()

            # -----------------------------------------------
            # HISTORICAL CRITIC RESULT
            # -----------------------------------------------

            critic_result = {
                "critic_id": critic_id,
                "problem_id": current_problem_id,
                "runtime": runtime,
                "auto_runtime": auto_runtime,
                "action": action_item,
                "critique": critique_data,
                "status": "SUCCESS",
            }

            critic_results.append(critic_result)

            critic_records.append(critic_record.copy())

            # -----------------------------------------------
            # REMOVE FROM ACTIVE
            # -----------------------------------------------

            if critic_record in active_critics:
                active_critics.remove(critic_record)

            # -----------------------------------------------
            # CONDITIONAL QUEUE
            # -----------------------------------------------

            if critique.verdict == "CONDITIONAL":
                conditional_critics.append(
                    {
                        "critic_id": critic_id,
                        "problem_id": current_problem_id,
                        "runtime": runtime,
                        "auto_runtime": auto_runtime,
                        "action": action_item,
                        "critique": critique_data,
                        "advisor_strategy": target_strategy,
                    }
                )

            # -----------------------------------------------
            # CRITIC LOG
            # -----------------------------------------------

            logs.append(
                {
                    "agent": (f"Reality-Check Critic #{critic_id}"),
                    "runtime": runtime,
                    "auto_runtime": auto_runtime,
                    "problem_id": current_problem_id,
                    "prompt": {
                        "verified_context": critic_verified_context,
                        "advisor_strategy": target_strategy,
                        "target_action": action_item,
                    },
                    "output": critique_data,
                    "status": "SUCCESS",
                }
            )

        except Exception as e:
            # -----------------------------------------------
            # CRITIC FAILURE
            # -----------------------------------------------

            critic_results.append(
                {
                    "critic_id": critic_id,
                    "problem_id": current_problem_id,
                    "runtime": runtime,
                    "auto_runtime": auto_runtime,
                    "action": action_item,
                    "critique": None,
                    "status": "ERROR",
                    "error": str(e),
                }
            )

            critic_records.append(critic_record.copy())

            if critic_record in active_critics:
                active_critics.remove(critic_record)

            logs.append(
                {
                    "agent": (f"Reality-Check Critic #{critic_id}"),
                    "runtime": runtime,
                    "auto_runtime": auto_runtime,
                    "problem_id": current_problem_id,
                    "prompt": {
                        "verified_context": critic_verified_context,
                        "advisor_strategy": target_strategy,
                        "target_action": action_item,
                    },
                    "output": {"error": str(e)},
                    "status": "ERROR",
                }
            )

    # ========================================================
    # 10. RETURN CORE PIPELINE RESULT
    # ========================================================

    return {
        "response": advisor_response,
        "classification": classification,
        "needs_input": False,
        "verified_facts": verified_facts,
        "systematic_advice": systematic_advice,
        "critic_results": critic_results,
        "critic_records": critic_records,
        "conditional_critics": conditional_critics,
        "active_critics": active_critics,
        "critic_id_counter": critic_id_counter,
        "logs": logs,
    }
