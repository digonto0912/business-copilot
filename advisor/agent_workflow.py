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
    problem_identifier_agent,
    problem_explainer_agent,
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
# MAIN WORKFLOW
# ============================================================


def run_workflow(
    history,
    verified_facts_memory=None,
    runtime=1,
    auto_runtime=0,
    problem_stack=None,
    current_problem_index=0,
    critic_id_counter=0,
    active_critics=None,
    max_problem_depth=3,
):
    """
    Main architectural flow.

    runtime:
        Human-started architectural flow number.

    auto_runtime:
        Automatic architectural re-entry number inside
        the current runtime.

        IMPORTANT:
        This value is NOT incremented for individual agents.
        It only changes when the entire architecture is
        automatically started again.

    problem_stack:
        Current parent/child problem hierarchy.

    current_problem_index:
        Index of the currently active problem.

    critic_id_counter:
        Global sequential critic ID.

    active_critics:
        Critics that are currently active/unresolved.

    max_problem_depth:
        Maximum allowed child-problem depth.
    """

    # ========================================================
    # NORMALIZE MEMORY
    # ========================================================

    if verified_facts_memory is None:
        verified_facts_memory = []

    if problem_stack is None:
        problem_stack = []

    if active_critics is None:
        active_critics = []

    # ========================================================
    # 1. LATEST USER INPUT
    # ========================================================

    user_messages = [content for role, content in history if role == "user"]

    latest_user_input = user_messages[-1] if user_messages else ""

    # ========================================================
    # 2. PROBLEM IDENTIFIER
    # ========================================================

    problem_identifier_debug = LLMDebugHandler()

    problem_identifier_result = None
    problem_identifier_status = "ERROR"

    try:
        problem_identifier_result = problem_identifier_agent.with_config(
            callbacks=[problem_identifier_debug]
        ).invoke({"user_input": latest_user_input})

        problem_identifier_status = "SUCCESS"

    except Exception as e:
        problem_identifier_debug.llm_output = {"error": str(e)}

    # ========================================================
    # 3. CREATE ROOT PROBLEM
    # ========================================================
    #
    # The first actual problem becomes problem 0.
    #
    # Q&A/context does not create a problem.
    #
    # ========================================================

    if (
        problem_identifier_result is not None
        and problem_identifier_result.classification == "PROBLEM"
        and not problem_stack
    ):
        root_problem = {
            "problem_id": 0,
            "parent_problem_id": None,
            "depth": 0,
            "problem": (problem_identifier_result.problem),
            "runtime": runtime,
            "auto_runtime": auto_runtime,
            "agent": "Advisor",
            "status": "ACTIVE",
            "solution": None,
        }

        problem_stack.append(root_problem)

        current_problem_index = 0

    # ========================================================
    # 4. CURRENT PROBLEM
    # ========================================================

    if problem_stack:
        current_problem = problem_stack[current_problem_index]

        current_problem_id = current_problem["problem_id"]

    else:
        current_problem = None
        current_problem_id = None

    # ========================================================
    # 5. ADVISOR
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
    # 6. FIND LAST TWO HUMAN MESSAGES
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
    # 7. VERIFIED FACTS
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
    # 8. CLASSIFIER
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
    # 9. DECISION
    # ========================================================

    needs_input = "NEEDS_INPUT" in classification

    # ========================================================
    # 10. SYSTEMATIC ADVICE
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
    # 11. VERIFIED FACTS MEMORY FOR CRITIC
    # ========================================================

    critic_verified_context = []

    if verified_facts_memory:
        critic_verified_context.extend(verified_facts_memory)

    if verified_facts is not None:
        critic_verified_context.append(verified_facts.model_dump())

    # ========================================================
    # 12. CRITICS
    # ========================================================

    critic_results = []

    critic_records = []

    critic_status = "SKIPPED"

    # --------------------------------------------------------
    # Conditional problems created in this run
    # --------------------------------------------------------

    child_problems_created = []

    if not needs_input and systematic_advice is not None:
        critic_status = "SUCCESS"

        advisor_strategy = systematic_advice.model_dump()

        all_actions = advisor_strategy.get(
            "prioritized_action_plan",
            [],
        )

        # ====================================================
        # ONE CRITIC PER ACTION
        # ====================================================

        for action_item in all_actions:
            # -----------------------------------------------
            # GLOBAL CRITIC ID
            # -----------------------------------------------

            critic_id_counter += 1

            critic_id = critic_id_counter

            # -----------------------------------------------
            # CRITIC RECORD
            # -----------------------------------------------

            critic_record = {
                "critic_id": critic_id,
                "problem_id": current_problem_id,
                "runtime": runtime,
                "auto_runtime": auto_runtime,
            }

            active_critics.append(critic_record)

            # -----------------------------------------------
            # TARGET STRATEGY
            # -----------------------------------------------

            target_strategy = {
                key: value
                for key, value in (advisor_strategy.items())
                if key != ("prioritized_action_plan")
            }

            # Only this one action is exposed
            # to the critic.

            target_strategy["prioritized_action_plan"] = [action_item]

            # -----------------------------------------------
            # RUN CRITIC
            # -----------------------------------------------

            try:
                critique = critic_agent.critique_action(
                    action_item=action_item,
                    verified_context=(critic_verified_context),
                    advisor_strategy=(target_strategy),
                )

                critique_data = critique.model_dump()

                # -------------------------------------------
                # SAVE CRITIC RESULT
                # -------------------------------------------

                critic_results.append(
                    {
                        "critic_id": critic_id,
                        "problem_id": current_problem_id,
                        "runtime": runtime,
                        "auto_runtime": auto_runtime,
                        "action": action_item,
                        "critique": critique_data,
                        "status": "SUCCESS",
                    }
                )

                # -------------------------------------------
                # HISTORICAL CRITIC RECORD
                # -------------------------------------------

                critic_records.append(critic_record.copy())

                # -------------------------------------------
                # REMOVE ACTIVE CRITIC
                # -------------------------------------------

                if critic_record in active_critics:
                    active_critics.remove(critic_record)

                # =================================================
                # CONDITIONAL CRITIC
                # =================================================

                if critique.verdict == "CONDITIONAL":
                    problem_explainer_debug = LLMDebugHandler()

                    problem_explainer_result = None

                    problem_explainer_status = "ERROR"

                    try:
                        # -----------------------------------------
                        # Problem Explainer input
                        # -----------------------------------------

                        problem_explainer_result = problem_explainer_agent.with_config(
                            callbacks=[problem_explainer_debug]
                        ).invoke(
                            {
                                "current_problem": (json_safe_problem(current_problem)),
                                "conditional_critic": critique_data,
                                "verified_facts": critic_verified_context,
                            }
                        )

                        problem_explainer_status = "SUCCESS"

                    except Exception as e:
                        problem_explainer_debug.llm_output = {"error": str(e)}

                    # ---------------------------------------------
                    # NEW PROBLEM
                    # ---------------------------------------------

                    if (
                        problem_explainer_result is not None
                        and problem_explainer_result.classification == "NEW_PROBLEM"
                        and problem_explainer_result.problem
                    ):
                        parent_depth = (
                            current_problem["depth"] if current_problem else -1
                        )

                        new_depth = parent_depth + 1

                        # -----------------------------------------
                        # DEPTH CONTROL
                        # -----------------------------------------

                        if new_depth <= max_problem_depth:
                            new_problem_id = len(problem_stack)

                            new_problem = {
                                "problem_id": new_problem_id,
                                "parent_problem_id": current_problem_id,
                                "depth": new_depth,
                                "problem": (problem_explainer_result.problem),
                                "runtime": runtime,
                                # IMPORTANT:
                                # auto_runtime is NOT
                                # incremented here.
                                #
                                # It increments only when
                                # the new problem actually
                                # causes automatic re-entry.
                                "auto_runtime": auto_runtime,
                                "agent": "Advisor",
                                "status": "PENDING_AUTO_RUN",
                                "solution": None,
                            }

                            problem_stack.append(new_problem)

                            child_problems_created.append(
                                {
                                    "problem": new_problem,
                                    "source_critic_id": critic_id,
                                    "problem_explainer": (
                                        problem_explainer_result.model_dump()
                                    ),
                                    "runtime": runtime,
                                    "auto_runtime": auto_runtime,
                                }
                            )

                        else:
                            # -------------------------------------
                            # Depth limit reached.
                            #
                            # Do not push another problem.
                            # -------------------------------------

                            child_problems_created.append(
                                {
                                    "problem": None,
                                    "source_critic_id": critic_id,
                                    "problem_explainer": (
                                        problem_explainer_result.model_dump()
                                    ),
                                    "runtime": runtime,
                                    "auto_runtime": auto_runtime,
                                    "status": "MAX_DEPTH_REACHED",
                                }
                            )

                    # -------------------------------------------
                    # Problem Explainer debug log
                    # -------------------------------------------

                    if problem_explainer_debug is not None:
                        # Save a temporary internal collection
                        # below through critic_results metadata.
                        #
                        # We don't change the critic result itself.
                        pass

            except Exception as e:
                critic_status = "ERROR"

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

    # ========================================================
    # 13. SAVE CURRENT PROBLEM STATE
    # ========================================================

    if (
        current_problem is not None
        and systematic_advice is not None
        and not needs_input
    ):
        current_problem["solution"] = systematic_advice.model_dump()

        current_problem["status"] = "SOLVED"

    # ========================================================
    # 14. DEBUG LOGS
    # ========================================================

    logs = []

    # --------------------------------------------------------
    # Problem Identifier
    # --------------------------------------------------------

    logs.append(
        {
            "agent": "Problem Identifier",
            "runtime": runtime,
            "auto_runtime": auto_runtime,
            "problem_id": current_problem_id,
            "prompt": problem_identifier_debug.llm_prompt,
            "output": problem_identifier_debug.llm_output,
            "status": problem_identifier_status,
        }
    )

    # --------------------------------------------------------
    # Advisor
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Verified Facts
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Classifier
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Systematic Advice Converter
    # --------------------------------------------------------

    if systematic_advice_debug is not None:
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

    # --------------------------------------------------------
    # Critic Logs
    # --------------------------------------------------------

    for critic_result in critic_results:
        logs.append(
            {
                "agent": (f"Reality-Check Critic #{critic_result['critic_id']}"),
                "runtime": critic_result["runtime"],
                "auto_runtime": critic_result["auto_runtime"],
                "problem_id": critic_result["problem_id"],
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
    # 15. RETURN
    # ========================================================

    return {
        "response": advisor_response,
        "classification": classification,
        "needs_input": needs_input,
        "verified_facts": verified_facts,
        "systematic_advice": systematic_advice,
        "critic_results": critic_results,
        "critic_records": critic_records,
        "active_critics": active_critics,
        "problem_stack": problem_stack,
        "current_problem_index": current_problem_index,
        "critic_id_counter": critic_id_counter,
        "child_problems_created": child_problems_created,
        "logs": logs,
    }


# ============================================================
# SMALL JSON-SAFE HELPER
# ============================================================


def json_safe_problem(problem):
    """
    Return a plain dict for the Problem Explainer.
    """

    if problem is None:
        return {}

    return dict(problem)
