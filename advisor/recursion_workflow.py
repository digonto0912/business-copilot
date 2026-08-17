# recursion_workflow.py

import json

from langchain_core.callbacks import BaseCallbackHandler

from agent_workflow import (
    run_core_workflow,
)

from agents import (
    problem_identifier_agent,
    problem_explainer_agent,
)


# ============================================================
# LLM DEBUG CALLBACK
# ============================================================


class RecursionDebugHandler(BaseCallbackHandler):
    """
    Captures Problem Identifier /
    Problem Explainer prompts and outputs.
    """

    def __init__(self):

        self.llm_prompt = None
        self.llm_output = None

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
# PROBLEM IDENTIFIER
# ============================================================


def identify_problem(
    user_input,
):

    debug = RecursionDebugHandler()

    try:
        result = problem_identifier_agent.with_config(callbacks=[debug]).invoke(
            {"user_input": user_input}
        )

        return {
            "result": result,
            "debug": debug,
            "status": "SUCCESS",
        }

    except Exception as e:
        debug.llm_output = {"error": str(e)}

        return {
            "result": None,
            "debug": debug,
            "status": "ERROR",
        }


# ============================================================
# PROBLEM EXPLAINER
# ============================================================


def explain_problem(
    current_problem,
    conditional_critic,
    verified_facts,
):

    debug = RecursionDebugHandler()

    try:
        result = problem_explainer_agent.with_config(callbacks=[debug]).invoke(
            {
                "current_problem": json.dumps(
                    current_problem,
                    indent=2,
                    ensure_ascii=False,
                ),
                "conditional_critic": json.dumps(
                    conditional_critic,
                    indent=2,
                    ensure_ascii=False,
                ),
                "verified_facts": json.dumps(
                    verified_facts,
                    indent=2,
                    ensure_ascii=False,
                ),
            }
        )

        return {
            "result": result,
            "debug": debug,
            "status": "SUCCESS",
        }

    except Exception as e:
        debug.llm_output = {"error": str(e)}

        return {
            "result": None,
            "debug": debug,
            "status": "ERROR",
        }


# ============================================================
# RECURSIVE ARCHITECTURAL FLOW
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
    Controls the recursive architecture.

    It does NOT implement the core agent pipeline itself.

    It controls:

        runtime
        auto_runtime
        problem stack
        current problem
        Problem Identifier
        Problem Explainer
        child creation
        depth limit
        automatic re-entry
        child completion
        parent restoration
    """

    # ========================================================
    # NORMALIZE
    # ========================================================

    if verified_facts_memory is None:
        verified_facts_memory = []

    if problem_stack is None:
        problem_stack = []

    if active_critics is None:
        active_critics = []

    # ========================================================
    # MEMORY
    # ========================================================

    all_logs = []

    all_critic_results = []

    all_critic_records = []

    all_child_problems = []

    # ========================================================
    # FLOW VARIABLES
    # ========================================================

    current_auto_runtime = auto_runtime

    current_history = list(history)

    # ========================================================
    # HUMAN FLOW
    # ========================================================

    human_user_messages = [content for role, content in history if role == "user"]

    latest_user_input = human_user_messages[-1] if human_user_messages else ""

    # ========================================================
    # PROBLEM IDENTIFIER
    # ========================================================
    #
    # Only the initial human-started flow gets Problem
    # Identifier.
    #
    # Automatic child flows already have their problem.
    #
    # ========================================================

    if current_auto_runtime == 0:
        identifier = identify_problem(latest_user_input)

        identifier_result = identifier["result"]

        identifier_log = {
            "agent": "Problem Identifier",
            "runtime": runtime,
            "auto_runtime": current_auto_runtime,
            "problem_id": (
                problem_stack[current_problem_index]["problem_id"]
                if problem_stack
                else None
            ),
            "prompt": identifier["debug"].llm_prompt,
            "output": identifier["debug"].llm_output,
            "status": identifier["status"],
        }

        all_logs.append(identifier_log)

        # ----------------------------------------------------
        # CREATE ROOT PROBLEM
        # ----------------------------------------------------

        if (
            identifier_result is not None
            and identifier_result.classification == "PROBLEM"
            and not problem_stack
        ):
            root_problem = {
                "problem_id": 0,
                "parent_problem_id": None,
                "depth": 0,
                "problem": identifier_result.problem,
                "runtime": runtime,
                "auto_runtime": 0,
                "agent": "Advisor",
                "status": "ACTIVE",
                "solution": None,
            }

            problem_stack.append(root_problem)

            current_problem_index = 0

            # Link the debug record to the problem created by this exact call.
            identifier_log["problem_id"] = root_problem["problem_id"]

    # ========================================================
    # RECURSION LOOP
    # ========================================================
    #
    # The core pipeline executes one complete flow.
    #
    # If a conditional critic creates a new problem:
    #
    #     push child
    #     auto_runtime += 1
    #     build child history
    #     run core pipeline again
    #
    # ========================================================

    while True:
        # ====================================================
        # CURRENT PROBLEM
        # ====================================================

        if problem_stack:
            current_problem = problem_stack[current_problem_index]

            current_problem_id = current_problem["problem_id"]

        else:
            current_problem = None
            current_problem_id = None

        # ====================================================
        # RUN CORE PIPELINE
        # ====================================================

        result = run_core_workflow(
            history=current_history,
            verified_facts_memory=(verified_facts_memory),
            runtime=runtime,
            auto_runtime=(current_auto_runtime),
            current_problem=(current_problem),
            current_problem_id=(current_problem_id),
            critic_id_counter=(critic_id_counter),
            active_critics=(active_critics),
        )

        # ====================================================
        # MERGE CORE RESULTS
        # ====================================================

        all_logs.extend(result["logs"])

        all_critic_results.extend(result["critic_results"])

        all_critic_records.extend(result["critic_records"])

        critic_id_counter = result["critic_id_counter"]

        active_critics = result["active_critics"]

        # ----------------------------------------------------
        # VERIFIED FACTS
        # ----------------------------------------------------
        #
        # The current core result becomes the latest verified
        # facts memory item.
        #
        # The caller will persist the returned memory.
        #
        # ----------------------------------------------------

        current_verified_facts = []

        if verified_facts_memory:
            current_verified_facts.extend(verified_facts_memory)

        if result["verified_facts"] is not None:
            current_verified_facts.append(result["verified_facts"].model_dump())

        # ====================================================
        # USER STILL NEEDS TO ANSWER
        # ====================================================

        if result["needs_input"]:
            return {
                **result,
                "logs": all_logs,
                "critic_results": all_critic_results,
                "critic_records": all_critic_records,
                "active_critics": active_critics,
                "problem_stack": problem_stack,
                "current_problem_index": current_problem_index,
                "critic_id_counter": critic_id_counter,
                "verified_facts_memory": current_verified_facts,
                "auto_runtime": current_auto_runtime,
            }

        # ====================================================
        # CURRENT ADVICE
        # ====================================================

        systematic_advice = result["systematic_advice"]

        if systematic_advice is None:
            return {
                **result,
                "logs": all_logs,
                "critic_results": all_critic_results,
                "critic_records": all_critic_records,
                "active_critics": active_critics,
                "problem_stack": problem_stack,
                "current_problem_index": current_problem_index,
                "critic_id_counter": critic_id_counter,
                "verified_facts_memory": current_verified_facts,
                "auto_runtime": current_auto_runtime,
            }

        # ====================================================
        # CONDITIONAL CRITICS
        # ====================================================

        conditional_critics = result["conditional_critics"]

        # ====================================================
        # NO CONDITIONAL CRITICS
        # ====================================================

        if not conditional_critics:
            # -----------------------------------------------
            # Solve current problem
            # -----------------------------------------------

            if current_problem is not None:
                current_problem["solution"] = systematic_advice.model_dump()

                current_problem["status"] = "SOLVED"

            # -----------------------------------------------
            # Child problem completed?
            # -----------------------------------------------

            if len(problem_stack) > 1:
                finished_problem = problem_stack.pop()

                current_problem_index = len(problem_stack) - 1

                parent_problem = problem_stack[current_problem_index]

                parent_problem["status"] = "ACTIVE"

                # -------------------------------------------
                # Attach child result to parent
                # -------------------------------------------

                if "child_solutions" not in parent_problem:
                    parent_problem["child_solutions"] = []

                parent_problem["child_solutions"].append(
                    {
                        "problem": finished_problem["problem"],
                        "solution": finished_problem["solution"],
                        "problem_id": finished_problem["problem_id"],
                    }
                )

                # Parent has resumed, but we do not
                # automatically ask the advisor anything
                # until another explicit recursive problem
                # is generated.

            # -----------------------------------------------
            # Root solved
            # -----------------------------------------------

            else:
                if current_problem is not None:
                    current_problem["status"] = "SOLVED"

            return {
                **result,
                "logs": all_logs,
                "critic_results": all_critic_results,
                "critic_records": all_critic_records,
                "active_critics": active_critics,
                "problem_stack": problem_stack,
                "current_problem_index": current_problem_index,
                "critic_id_counter": critic_id_counter,
                "verified_facts_memory": current_verified_facts,
                "auto_runtime": current_auto_runtime,
            }

        # ====================================================
        # PROCESS CONDITIONAL CRITICS
        # ====================================================
        #
        # For now we use the first conditional critic to create
        # one child problem.
        #
        # Later this can become a proper queue.
        #
        # ====================================================

        created_child = None

        for conditional in conditional_critics:
            explainer = explain_problem(
                current_problem=(current_problem),
                conditional_critic=(conditional["critique"]),
                verified_facts=(current_verified_facts),
            )

            # -----------------------------------------------
            # SAVE EXPLAINER LOG
            # -----------------------------------------------

            explainer_log = {
                "agent": "Problem Explainer",
                "runtime": runtime,
                "auto_runtime": current_auto_runtime,
                "problem_id": current_problem_id,
                "prompt": explainer["debug"].llm_prompt,
                "output": explainer["debug"].llm_output,
                "status": explainer["status"],
            }

            all_logs.append(explainer_log)

            explainer_result = explainer["result"]

            if explainer_result is None:
                continue

            if explainer_result.classification != "NEW_PROBLEM":
                continue

            if not (explainer_result.problem):
                continue

            # -----------------------------------------------
            # DEPTH
            # -----------------------------------------------

            parent_depth = current_problem.get(
                "depth",
                0,
            )

            new_depth = parent_depth + 1

            if new_depth > max_problem_depth:
                # Max depth reached.
                # Do not create another child.

                all_child_problems.append(
                    {
                        "status": "MAX_DEPTH_REACHED",
                        "parent_problem_id": current_problem_id,
                        "runtime": runtime,
                        "auto_runtime": current_auto_runtime,
                        "problem": explainer_result.problem,
                    }
                )

                continue

            # -----------------------------------------------
            # NEW PROBLEM ID
            # -----------------------------------------------

            new_problem_id = (
                max(
                    (problem["problem_id"] for problem in problem_stack),
                    default=-1,
                )
                + 1
            )

            # -----------------------------------------------
            # CREATE CHILD PROBLEM
            # -----------------------------------------------

            created_child = {
                "problem_id": new_problem_id,
                "parent_problem_id": current_problem_id,
                "depth": new_depth,
                "problem": explainer_result.problem,
                # This raw problem record is created by the Problem
                # Explainer call above, so it keeps that exact call's
                # execution metadata. The automatic Advisor call advances
                # auto_runtime separately below.
                "runtime": runtime,
                "auto_runtime": current_auto_runtime,
                "agent": "Advisor",
                "status": "ACTIVE",
                "solution": None,
                "source_critic_id": conditional["critic_id"],
            }

            # Link the debug record to the child problem created by this
            # exact Problem Explainer call.
            explainer_log["problem_id"] = new_problem_id

            # Parent waits.

            if current_problem is not None:
                current_problem["status"] = "WAITING_FOR_CHILD"

            # Push.

            problem_stack.append(created_child)

            current_problem_index = len(problem_stack) - 1

            all_child_problems.append(created_child)

            # Only one child problem at a time.
            break

        # ====================================================
        # NO CHILD CREATED
        # ====================================================

        if created_child is None:
            # Current problem is solved at this level.

            if current_problem is not None:
                current_problem["solution"] = systematic_advice.model_dump()

                current_problem["status"] = "SOLVED"

            return {
                **result,
                "logs": all_logs,
                "critic_results": all_critic_results,
                "critic_records": all_critic_records,
                "active_critics": active_critics,
                "problem_stack": problem_stack,
                "current_problem_index": current_problem_index,
                "critic_id_counter": critic_id_counter,
                "verified_facts_memory": current_verified_facts,
                "auto_runtime": current_auto_runtime,
                "child_problems": all_child_problems,
            }

        # ====================================================
        # AUTOMATIC RE-ENTRY
        # ====================================================

        current_auto_runtime += 1

        # ----------------------------------------------------
        # Build the automatic problem-solving input.
        # ----------------------------------------------------

        latest_conditional = conditional_critics[0]

        automatic_problem_payload = (
            "AUTOMATIC PROBLEM-SOLVING REQUEST\n\n"
            "NEW PROBLEM:\n"
            + created_child["problem"]
            + "\n\nSOURCE CRITIC:\n"
            + json.dumps(
                latest_conditional["critique"],
                indent=2,
                ensure_ascii=False,
            )
            + "\n\nFULL VERIFIED FACTS:\n"
            + json.dumps(
                current_verified_facts,
                indent=2,
                ensure_ascii=False,
            )
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # This is an automatic re-entry.
        # We do NOT create a new human runtime.
        #
        # ----------------------------------------------------

        current_history = list(history)

        current_history.append(
            (
                "user",
                automatic_problem_payload,
            )
        )

        # Continue the recursion loop.
