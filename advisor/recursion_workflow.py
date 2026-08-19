# recursion_workflow.py

import json
from collections import deque

from langchain_core.callbacks import BaseCallbackHandler

from agent_workflow import (
    repair_failed_action,
    run_core_workflow,
)
from agents import problem_explainer_agent, problem_identifier_agent, CriticAgent
from rate_limit import gemini_31_flash_lite_quota, get_quota_snapshot


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MAX_FAIL_REPAIR_ATTEMPTS = 3

critic_agent = CriticAgent()


# ============================================================
# DEBUG
# ============================================================


class RecursionDebugHandler(BaseCallbackHandler):
    """Captures Problem Identifier / Problem Explainer prompts and outputs."""

    def __init__(self):
        self.llm_prompt = None
        self.llm_output = None

    def on_chat_model_start(self, serialized, messages, **kwargs):
        try:
            prompt_messages = []
            for batch in messages:
                for message in batch:
                    prompt_messages.append(
                        {
                            "role": getattr(message, "type", "unknown"),
                            "content": message.content,
                        }
                    )
            self.llm_prompt = prompt_messages
        except Exception as e:
            self.llm_prompt = {"error": str(e)}

    def on_llm_end(self, response, **kwargs):
        try:
            generation = response.generations[0][0]
            if hasattr(generation, "message"):
                message = generation.message
                self.llm_output = {
                    "role": getattr(message, "type", "ai"),
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
# SINGLE-AGENT HELPERS
# ============================================================


def identify_problem(user_input):
    debug = RecursionDebugHandler()
    try:
        result = problem_identifier_agent.with_config(
            callbacks=[debug, gemini_31_flash_lite_quota]
        ).invoke({"user_input": user_input})
        return {"result": result, "debug": debug, "status": "SUCCESS"}
    except Exception as e:
        debug.llm_output = {"error": str(e)}
        return {"result": None, "debug": debug, "status": "ERROR"}


def explain_problem(current_problem, conditional_critic, verified_facts):
    debug = RecursionDebugHandler()
    try:
        result = problem_explainer_agent.with_config(
            callbacks=[debug, gemini_31_flash_lite_quota]
        ).invoke(
            {
                "current_problem": json.dumps(
                    current_problem, indent=2, ensure_ascii=False
                ),
                "conditional_critic": json.dumps(
                    conditional_critic, indent=2, ensure_ascii=False
                ),
                "verified_facts": json.dumps(
                    verified_facts, indent=2, ensure_ascii=False
                ),
            }
        )
        return {"result": result, "debug": debug, "status": "SUCCESS"}
    except Exception as e:
        debug.llm_output = {"error": str(e)}
        return {"result": None, "debug": debug, "status": "ERROR"}


# ============================================================
# TREE HELPERS
# ============================================================


def _iter_action_chain(action):
    """Yield an original action plus every FAIL repair revision."""
    yield action
    revision = action.get("repair_chain")
    while revision:
        yield revision
        revision = revision.get("next_revision")


def _iter_child_nodes(action):
    for action_version in _iter_action_chain(action):
        child = action_version.get("new_problem")
        if child:
            yield child


def _next_problem_id(problem_tree):
    ids = []

    def walk(node):
        if not isinstance(node, dict):
            return
        if isinstance(node.get("problem_id"), int):
            ids.append(node["problem_id"])
        for action in node.get("action_plans", []):
            for child in _iter_child_nodes(action):
                walk(child)

    for root in problem_tree:
        walk(root)

    return max(ids, default=-1) + 1


def _find_problem(problem_tree, problem_id):
    found = None

    def walk(node):
        nonlocal found
        if found is not None or not isinstance(node, dict):
            return
        if node.get("problem_id") == problem_id:
            found = node
            return
        for action in node.get("action_plans", []):
            for child in _iter_child_nodes(action):
                walk(child)

    for root in problem_tree:
        walk(root)

    return found


def _build_problem_node(
    problem_id,
    parent_problem_id,
    depth,
    problem,
    runtime,
    auto_runtime,
    source_critic_id=None,
):
    return {
        "problem_id": problem_id,
        "parent_problem_id": parent_problem_id,
        "depth": depth,
        "problem": problem,
        "runtime": runtime,
        "auto_runtime": auto_runtime,
        "agent": "Advisor",
        "status": "ACTIVE",
        "solution": None,
        "source_critic_id": source_critic_id,
        "action_plans": [],
    }


def _automatic_problem_history(base_history, child_problem, source_critic, verified_facts):
    payload = (
        "AUTOMATIC PROBLEM-SOLVING REQUEST\n\n"
        "NEW PROBLEM:\n"
        + child_problem["problem"]
        + "\n\nSOURCE CRITIC:\n"
        + json.dumps(source_critic, indent=2, ensure_ascii=False)
        + "\n\nFULL VERIFIED FACTS:\n"
        + json.dumps(verified_facts, indent=2, ensure_ascii=False)
    )

    child_history = list(base_history)
    child_history.append(("user", payload))
    return child_history


def _max_auto_runtime(problem_tree):
    maximum = 0

    def walk(node):
        nonlocal maximum
        if not isinstance(node, dict):
            return
        value = node.get("auto_runtime")
        if isinstance(value, int):
            maximum = max(maximum, value)
        for action in node.get("action_plans", []):
            for child in _iter_child_nodes(action):
                walk(child)

    for root in problem_tree:
        walk(root)

    return maximum


# ============================================================
# ACTION NODE HELPERS
# ============================================================


def _build_action_node(critic_result, runtime, auto_runtime):
    critique = critic_result.get("critique") or {}
    verdict = critique.get("verdict") if isinstance(critique, dict) else None

    action_node = {
        "action_plan": critic_result.get("action"),
        "critic_id": critic_result["critic_id"],
        "runtime": runtime,
        "auto_runtime": auto_runtime,
        "verdict": verdict,
        "verdict_reason": (
            critique.get("verdict_reason")
            if isinstance(critique, dict)
            else None
        ),
        "status": {
            "PASS": "PASSED",
            "CONDITIONAL": "CONDITIONAL",
            "FAIL": "FAILED",
        }.get(verdict, "CRITIC_ERROR"),
        "new_problem": None,
        "new_problem_classification": None,
        "new_problem_reason": None,
        "final_action_plan": critic_result.get("action") if verdict == "PASS" else None,
        "repair_attempts": 0,
        "repair_attempt_log": [],
        "repair_chain": None,
    }

    return action_node


def _allocate_critic_id(
    critic_id_counter,
    current_problem_id,
    runtime,
    auto_runtime,
    active_critics,
):
    critic_id_counter += 1
    critic_record = {
        "critic_id": critic_id_counter,
        "problem_id": current_problem_id,
        "runtime": runtime,
        "auto_runtime": auto_runtime,
    }
    active_critics.append(critic_record)
    return critic_id_counter, critic_record


def _critic_repaired_action(
    action_item,
    parent_strategy_context,
    verified_context,
    problem_id,
    runtime,
    auto_runtime,
    critic_id_counter,
    active_critics,
):
    critic_id_counter, critic_record = _allocate_critic_id(
        critic_id_counter=critic_id_counter,
        current_problem_id=problem_id,
        runtime=runtime,
        auto_runtime=auto_runtime,
        active_critics=active_critics,
    )

    target_strategy = dict(parent_strategy_context)
    target_strategy["prioritized_action_plan"] = [action_item]

    try:
        critique = critic_agent.critique_action(
            action_item=action_item,
            verified_context=verified_context,
            advisor_strategy=target_strategy,
        )
        critique_data = critique.model_dump()
        status = "SUCCESS"
        error = None
    except Exception as e:
        critique_data = None
        status = "ERROR"
        error = str(e)

    if critic_record in active_critics:
        active_critics.remove(critic_record)

    record = critic_record.copy()
    record["status"] = status
    if error:
        record["error"] = error

    log = {
        "agent": f"Reality-Check Critic #{critic_id_counter}",
        "runtime": runtime,
        "auto_runtime": auto_runtime,
        "problem_id": problem_id,
        "critic_id": critic_id_counter,
        "prompt": {
            "verified_context": verified_context,
            "advisor_strategy": target_strategy,
            "target_action": action_item,
        },
        "output": critique_data if critique_data is not None else {"error": error},
        "status": status,
    }

    return (
        {
            "critic_id": critic_id_counter,
            "problem_id": problem_id,
            "runtime": runtime,
            "auto_runtime": auto_runtime,
            "action": action_item,
            "critique": critique_data,
            "status": status,
            **({"error": error} if error else {}),
        },
        record,
        log,
        critic_id_counter,
    )


def _attach_conditional_result(
    action_version,
    current_problem,
    conditional_critic,
    verified_facts,
    current_history,
    runtime,
    next_auto_runtime,
    max_problem_depth,
    problem_tree,
    all_child_problems,
):
    """Run Problem Explainer for a CONDITIONAL action only."""

    explainer = explain_problem(
        current_problem=current_problem,
        conditional_critic=conditional_critic,
        verified_facts=verified_facts,
    )

    result = explainer["result"]
    action_version["new_problem"] = None
    action_version["new_problem_classification"] = None
    action_version["new_problem_reason"] = None

    if result is None:
        action_version["status"] = "CONDITIONAL_EXPLAINER_ERROR"
        return {
            "explainer": explainer,
            "child": None,
            "history": None,
            "next_auto_runtime": next_auto_runtime,
        }

    action_version["new_problem_classification"] = result.classification
    action_version["new_problem_reason"] = result.reason

    if result.classification != "NEW_PROBLEM" or not result.problem:
        action_version["status"] = "CONDITIONAL_NO_NEW_PROBLEM"
        return {
            "explainer": explainer,
            "child": None,
            "history": None,
            "next_auto_runtime": next_auto_runtime,
        }

    # max_problem_depth is user-facing as TOTAL PROBLEM LAYERS,
    # including the root (depth 0). Therefore a setting of 3 allows
    # depths 0, 1, and 2 only.
    max_allowed_depth = max(0, max_problem_depth - 1)
    new_depth = current_problem.get("depth", 0) + 1
    if new_depth > max_allowed_depth:
        action_version["new_problem_classification"] = "MAX_DEPTH_REACHED"
        action_version["new_problem_reason"] = (
            "A new problem was identified, but the configured maximum problem depth was reached."
        )
        action_version["status"] = "CONDITIONAL_MAX_DEPTH_REACHED"
        return {
            "explainer": explainer,
            "child": None,
            "history": None,
            "next_auto_runtime": next_auto_runtime,
        }

    new_problem_id = _next_problem_id(problem_tree)
    next_auto_runtime += 1

    child_node = _build_problem_node(
        problem_id=new_problem_id,
        parent_problem_id=current_problem["problem_id"],
        depth=new_depth,
        problem=result.problem,
        runtime=runtime,
        auto_runtime=next_auto_runtime,
        source_critic_id=action_version["critic_id"],
    )

    action_version["new_problem"] = child_node
    action_version["status"] = "CONDITIONAL_NEW_PROBLEM"
    all_child_problems.append(child_node)

    child_history = _automatic_problem_history(
        base_history=current_history,
        child_problem=child_node,
        source_critic=conditional_critic,
        verified_facts=verified_facts,
    )

    return {
        "explainer": explainer,
        "child": child_node,
        "history": child_history,
        "next_auto_runtime": next_auto_runtime,
    }


def _append_repair_revision(parent_action, revision):
    if parent_action.get("repair_chain") is None:
        parent_action["repair_chain"] = revision
        return

    cursor = parent_action["repair_chain"]
    while cursor.get("next_revision") is not None:
        cursor = cursor["next_revision"]
    cursor["next_revision"] = revision


# ============================================================
# MAIN WORKFLOW
# ============================================================


def run_workflow(
    history,
    verified_facts_memory=None,
    runtime=1,
    auto_runtime=0,
    problem_tree=None,
    critic_id_counter=0,
    active_critics=None,
    max_problem_depth=3,
    max_fail_repair_attempts=DEFAULT_MAX_FAIL_REPAIR_ATTEMPTS,
    problem_stack=None,
    current_problem_index=0,
):
    """
    BFS orchestration with three critic verdict branches:

    PASS:
        keep the action unchanged.

    CONDITIONAL:
        run Problem Explainer and, if it finds a distinct problem,
        create a normal child Problem node.

    FAIL:
        do NOT run Problem Explainer. Instead repeatedly ask the Advisor
        to repair the SAME action using the parent problem, failed action,
        critic feedback, and verified facts. Each repaired version is
        re-criticised. The repair history is stored as a linked list.
        A bounded retry count prevents endless Advisor calls.
    """

    if verified_facts_memory is None:
        verified_facts_memory = []
    if active_critics is None:
        active_critics = []

    if problem_tree is None:
        problem_tree = []
        if problem_stack:
            problem_tree.extend(problem_stack)
            for node in problem_tree:
                node.setdefault("action_plans", [])

    all_logs = []
    all_critic_results = []
    all_critic_records = []
    all_child_problems = []

    current_verified_facts = list(verified_facts_memory)
    current_auto_runtime = auto_runtime
    next_auto_runtime = max(current_auto_runtime, _max_auto_runtime(problem_tree))

    # ------------------------------------------------------------
    # Initial user turn: identify/create the root problem.
    # ------------------------------------------------------------
    human_user_messages = [content for role, content in history if role == "user"]
    latest_user_input = human_user_messages[-1] if human_user_messages else ""

    if not problem_tree and current_auto_runtime == 0:
        identifier = identify_problem(latest_user_input)
        identifier_result = identifier["result"]

        all_logs.append(
            {
                "agent": "Problem Identifier",
                "runtime": runtime,
                "auto_runtime": current_auto_runtime,
                "problem_id": None,
                "prompt": identifier["debug"].llm_prompt,
                "output": identifier["debug"].llm_output,
                "status": identifier["status"],
                "quota": get_quota_snapshot("gemini-3.1-flash-lite"),
            }
        )

        if (
            identifier_result is not None
            and identifier_result.classification == "PROBLEM"
        ):
            root_problem = _build_problem_node(
                problem_id=0,
                parent_problem_id=None,
                depth=0,
                problem=identifier_result.problem,
                runtime=runtime,
                auto_runtime=0,
            )
            problem_tree.append(root_problem)

    if not problem_tree:
        return {
            "response": "",
            "classification": None,
            "needs_input": True,
            "verified_facts": None,
            "systematic_advice": None,
            "critic_results": [],
            "critic_records": [],
            "active_critics": active_critics,
            "logs": all_logs,
            "problem_tree": problem_tree,
            "problem_stack": problem_tree,
            "current_problem_index": 0,
            "critic_id_counter": critic_id_counter,
            "verified_facts_memory": current_verified_facts,
            "auto_runtime": current_auto_runtime,
            "child_problems": all_child_problems,
        }

    queue = deque()
    root = problem_tree[0]
    queue.append((root, list(history)))

    human_turn_result = None

    while queue:
        current_problem, current_history = queue.popleft()
        current_problem_id = current_problem["problem_id"]
        current_auto_runtime = current_problem["auto_runtime"]

        if current_problem.get("status") == "SOLVED":
            continue

        extract_verified_facts = human_turn_result is None

        result = run_core_workflow(
            history=current_history,
            verified_facts_memory=current_verified_facts,
            runtime=runtime,
            auto_runtime=current_auto_runtime,
            current_problem=current_problem,
            current_problem_id=current_problem_id,
            critic_id_counter=critic_id_counter,
            active_critics=active_critics,
            extract_verified_facts=extract_verified_facts,
        )

        if human_turn_result is None:
            human_turn_result = result

        all_logs.extend(result["logs"])
        all_critic_results.extend(result["critic_results"])
        all_critic_records.extend(result["critic_records"])
        critic_id_counter = result["critic_id_counter"]
        active_critics = result["active_critics"]

        if result["verified_facts"] is not None:
            current_verified_facts.append(result["verified_facts"].model_dump())

        # Automatic child problems must never abort the entire BFS run.
        # A child may hit NEEDS_INPUT or an agent error/empty response;
        # record that child as a terminal execution state and continue
        # processing the rest of the queue. The original human-turn result
        # remains the response returned to the UI.
        if result["needs_input"]:
            current_problem["status"] = "AUTO_INPUT_REQUIRED"
            all_logs.append(
                {
                    "agent": "Workflow Controller",
                    "runtime": current_problem["runtime"],
                    "auto_runtime": current_problem["auto_runtime"],
                    "problem_id": current_problem_id,
                    "prompt": None,
                    "output": {
                        "reason": (
                            "Automatic child problem requested additional user input; "
                            "workflow did not stop."
                        ),
                        "advisor_response": result.get("response"),
                    },
                    "status": "AUTO_INPUT_REQUIRED",
                }
            )
            continue

        systematic_advice = result["systematic_advice"]
        if systematic_advice is None:
            current_problem["status"] = "ERROR"
            all_logs.append(
                {
                    "agent": "Workflow Controller",
                    "runtime": current_problem["runtime"],
                    "auto_runtime": current_problem["auto_runtime"],
                    "problem_id": current_problem_id,
                    "prompt": None,
                    "output": {
                        "reason": (
                            "This problem's core workflow failed or returned an empty/invalid "
                            "structured strategy. The BFS queue continues with remaining problems."
                        ),
                        "advisor_response": result.get("response"),
                    },
                    "status": "CHILD_WORKFLOW_ERROR",
                }
            )
            continue

        verified_context = current_verified_facts
        advisor_strategy = systematic_advice.model_dump()

        # ========================================================
        # BUILD INITIAL ACTION NODES
        # ========================================================

        current_problem["action_plans"] = []
        initial_action_nodes = []

        for critic_result in result["critic_results"]:
            action_node = _build_action_node(
                critic_result=critic_result,
                runtime=current_problem["runtime"],
                auto_runtime=current_problem["auto_runtime"],
            )
            current_problem["action_plans"].append(action_node)
            initial_action_nodes.append((critic_result, action_node))

        # ========================================================
        # PROCESS EACH VERDICT BRANCH
        # ========================================================

        next_children = []

        for critic_result, action_node in initial_action_nodes:
            critique = critic_result.get("critique")
            verdict = (
                critique.get("verdict")
                if isinstance(critique, dict)
                else None
            )

            # ----------------------------------------------------
            # CRITIC ERROR
            # ----------------------------------------------------
            if verdict is None:
                continue

            # ----------------------------------------------------
            # PASS: keep original action unchanged.
            # ----------------------------------------------------
            if verdict == "PASS":
                action_node["status"] = "PASSED"
                action_node["final_action_plan"] = action_node["action_plan"]
                continue

            # ----------------------------------------------------
            # CONDITIONAL: Problem Explainer ONLY here.
            # ----------------------------------------------------
            if verdict == "CONDITIONAL":
                explainer_result = _attach_conditional_result(
                    action_version=action_node,
                    current_problem=current_problem,
                    conditional_critic=critique,
                    verified_facts=verified_context,
                    current_history=current_history,
                    runtime=runtime,
                    next_auto_runtime=next_auto_runtime,
                    max_problem_depth=max_problem_depth,
                    problem_tree=problem_tree,
                    all_child_problems=all_child_problems,
                )

                explainer = explainer_result["explainer"]
                next_auto_runtime = explainer_result["next_auto_runtime"]

                all_logs.append(
                    {
                        "agent": "Problem Explainer",
                        "runtime": current_problem["runtime"],
                        "auto_runtime": current_problem["auto_runtime"],
                        "problem_id": current_problem_id,
                        "critic_id": critic_result["critic_id"],
                        "prompt": explainer["debug"].llm_prompt,
                        "output": explainer["debug"].llm_output,
                        "status": explainer["status"],
                        "quota": get_quota_snapshot("gemini-3.1-flash-lite"),
                    }
                )

                child = explainer_result["child"]
                if child is not None:
                    next_children.append(
                        (
                            child,
                            explainer_result["history"],
                        )
                    )

                continue

            # ----------------------------------------------------
            # FAIL: repair this ONE action with Advisor, then re-critic.
            # ----------------------------------------------------
            if verdict == "FAIL":
                action_node["status"] = "REPAIRING"
                current_action = action_node["action_plan"]
                current_critic = critique
                repair_history = []
                previous_version = action_node
                repaired = False

                for attempt_number in range(
                    1,
                    max_fail_repair_attempts + 1,
                ):
                    action_node["repair_attempts"] = attempt_number - 1

                    # Check the bound BEFORE making the Advisor call.
                    if attempt_number > max_fail_repair_attempts:
                        break

                    # --------------------------------------------
                    # AUTO ADVISOR: ONE REPAIRED ACTION
                    # --------------------------------------------
                    repair_result = repair_failed_action(
                        parent_problem=current_problem,
                        failed_action=current_action,
                        critic_feedback=current_critic,
                        verified_facts=verified_context,
                        repair_history=repair_history,
                        runtime=current_problem["runtime"],
                        auto_runtime=current_problem["auto_runtime"],
                        problem_id=current_problem_id,
                    )

                    repair_record = {
                        "attempt": attempt_number,
                        "failed_action": current_action,
                        "critic_feedback": current_critic,
                        "status": repair_result["status"],
                        "repaired_action": None,
                        "critic_id": None,
                        "critic": None,
                    }

                    all_logs.append(
                        {
                            "agent": "Failed Action Repair Advisor",
                            "runtime": current_problem["runtime"],
                            "auto_runtime": current_problem["auto_runtime"],
                            "problem_id": current_problem_id,
                            "repair_attempt": attempt_number,
                            "prompt": repair_result["debug"].llm_prompt,
                            "output": repair_result["debug"].llm_output,
                            "status": repair_result["status"],
                            "quota": get_quota_snapshot("gemma-4-31b-it"),
                        }
                    )

                    # --------------------------------------------
                    # REPAIR LLM FAILED
                    # --------------------------------------------
                    if repair_result["result"] is None:
                        repair_record["error"] = repair_result["debug"].llm_output
                        repair_history.append(repair_record)
                        action_node["repair_attempt_log"] = list(repair_history)
                        continue

                    repaired_action = repair_result["result"].model_dump()
                    repair_record["repaired_action"] = repaired_action

                    # --------------------------------------------
                    # RE-CRITIC THE REPAIRED ACTION
                    # --------------------------------------------
                    (
                        repaired_critic_result,
                        repaired_critic_record,
                        repaired_critic_log,
                        critic_id_counter,
                    ) = _critic_repaired_action(
                        action_item=repaired_action,
                        parent_strategy_context=advisor_strategy,
                        verified_context=verified_context,
                        problem_id=current_problem_id,
                        runtime=current_problem["runtime"],
                        auto_runtime=current_problem["auto_runtime"],
                        critic_id_counter=critic_id_counter,
                        active_critics=active_critics,
                    )

                    all_critic_results.append(repaired_critic_result)
                    all_critic_records.append(repaired_critic_record)
                    all_logs.append(repaired_critic_log)

                    revised_critique = repaired_critic_result.get("critique")
                    revised_verdict = (
                        revised_critique.get("verdict")
                        if isinstance(revised_critique, dict)
                        else None
                    )

                    revision = {
                        "attempt": attempt_number,
                        "action_plan": repaired_action,
                        "critic_id": repaired_critic_result["critic_id"],
                        "verdict": revised_verdict,
                        "verdict_reason": (
                            revised_critique.get("verdict_reason")
                            if isinstance(revised_critique, dict)
                            else None
                        ),
                        "status": "REPAIR_FAILED",
                        "new_problem": None,
                        "new_problem_classification": None,
                        "new_problem_reason": None,
                        "next_revision": None,
                        "repair_record": repair_record,
                    }

                    _append_repair_revision(previous_version, revision)
                    repair_history.append(repair_record | {
                        "critic_id": repaired_critic_result["critic_id"],
                        "critic": revised_critique,
                        "verdict": revised_verdict,
                    })
                    action_node["repair_attempt_log"] = list(repair_history)

                    action_node["repair_attempts"] = attempt_number

                    # --------------------------------------------
                    # REPAIRED ACTION PASSES
                    # --------------------------------------------
                    if revised_verdict == "PASS":
                        revision["status"] = "PASSED"
                        action_node["status"] = "REPAIRED_AND_PASSED"
                        action_node["final_action_plan"] = repaired_action
                        repaired = True
                        break

                    # --------------------------------------------
                    # REPAIRED ACTION IS CONDITIONAL
                    # --------------------------------------------
                    if revised_verdict == "CONDITIONAL":
                        explainer_result = _attach_conditional_result(
                            action_version=revision,
                            current_problem=current_problem,
                            conditional_critic=revised_critique,
                            verified_facts=verified_context,
                            current_history=current_history,
                            runtime=runtime,
                            next_auto_runtime=next_auto_runtime,
                            max_problem_depth=max_problem_depth,
                            problem_tree=problem_tree,
                            all_child_problems=all_child_problems,
                        )

                        explainer = explainer_result["explainer"]
                        next_auto_runtime = explainer_result["next_auto_runtime"]

                        all_logs.append(
                            {
                                "agent": "Problem Explainer",
                                "runtime": current_problem["runtime"],
                                "auto_runtime": current_problem["auto_runtime"],
                                "problem_id": current_problem_id,
                                "critic_id": repaired_critic_result["critic_id"],
                                "repair_attempt": attempt_number,
                                "prompt": explainer["debug"].llm_prompt,
                                "output": explainer["debug"].llm_output,
                                "status": explainer["status"],
                                "quota": get_quota_snapshot("gemini-3.1-flash-lite"),
                            }
                        )

                        child = explainer_result["child"]
                        if child is not None:
                            next_children.append(
                                (
                                    child,
                                    explainer_result["history"],
                                )
                            )

                        revision["status"] = (
                            "CONDITIONAL_NEW_PROBLEM"
                            if child is not None
                            else "CONDITIONAL_NO_NEW_PROBLEM"
                        )
                        action_node["status"] = "REPAIRED_CONDITIONAL"
                        action_node["final_action_plan"] = repaired_action
                        repaired = True
                        break

                    # --------------------------------------------
                    # REPAIRED ACTION IS STILL FAIL
                    # --------------------------------------------
                    if revised_verdict == "FAIL":
                        revision["status"] = "REPAIR_FAILED"
                        previous_version = revision
                        current_action = repaired_action
                        current_critic = revised_critique
                        continue

                    # Critic error: consume this attempt, log it, and try again.
                    revision["status"] = "CRITIC_ERROR"
                    previous_version = revision
                    current_action = repaired_action
                    current_critic = revised_critique or {
                        "verdict": None,
                        "verdict_reason": repaired_critic_result.get("error"),
                    }

                # ------------------------------------------------
                # REPAIR LIMIT REACHED / EXHAUSTED
                # ------------------------------------------------
                if not repaired:
                    action_node["status"] = "REJECTED"
                    if action_node.get("repair_chain") is not None:
                        cursor = action_node["repair_chain"]
                        while cursor.get("next_revision") is not None:
                            cursor = cursor["next_revision"]
                        cursor["status"] = "REJECTED"
                    action_node["repair_rejection_reason"] = (
                        "The failed action exceeded the configured maximum number "
                        "of Advisor repair attempts and was not accepted."
                    )
                    action_node["final_action_plan"] = None

        # Save the original strategy while action nodes hold branch outcomes.
        current_problem["solution"] = advisor_strategy
        current_problem["status"] = (
            "WAITING_FOR_CHILDREN" if next_children else "SOLVED"
        )

        # BFS barrier: all initial critics plus all fail-repair chains and
        # conditional explainers for this problem finish before children queue.
        for child in next_children:
            queue.append(child)

    # ============================================================
    # FINALIZE TREE STATUSES
    # ============================================================

    def finalize(node):
        unresolved = False
        for action in node.get("action_plans", []):
            for child in _iter_child_nodes(action):
                finalize(child)
                if child.get("status") not in {
                    "SOLVED",
                    "ERROR",
                    "WAITING_FOR_USER",
                }:
                    unresolved = True
        if node.get("status") == "WAITING_FOR_CHILDREN" and not unresolved:
            node["status"] = "SOLVED"

    for root_node in problem_tree:
        finalize(root_node)

    if human_turn_result is None:
        human_turn_result = {
            "response": "",
            "classification": None,
            "needs_input": False,
            "verified_facts": None,
            "systematic_advice": None,
        }

    return {
        **human_turn_result,
        "logs": all_logs,
        "critic_results": all_critic_results,
        "critic_records": all_critic_records,
        "active_critics": active_critics,
        "problem_tree": problem_tree,
        "problem_stack": problem_tree,
        "current_problem_index": 0,
        "critic_id_counter": critic_id_counter,
        "verified_facts_memory": current_verified_facts,
        "auto_runtime": current_auto_runtime,
        "child_problems": all_child_problems,
    }
