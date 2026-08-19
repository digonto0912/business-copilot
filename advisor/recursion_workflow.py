# recursion_workflow.py

import json
from collections import deque

from langchain_core.callbacks import BaseCallbackHandler

from agent_workflow import run_core_workflow
from agents import problem_identifier_agent, problem_explainer_agent
from rate_limit import gemini_31_flash_lite_quota, get_quota_snapshot


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


def identify_problem(user_input):
    debug = RecursionDebugHandler()
    try:
        result = problem_identifier_agent.with_config(callbacks=[debug, gemini_31_flash_lite_quota]).invoke(
            {"user_input": user_input}
        )
        return {"result": result, "debug": debug, "status": "SUCCESS"}
    except Exception as e:
        debug.llm_output = {"error": str(e)}
        return {"result": None, "debug": debug, "status": "ERROR"}


def explain_problem(current_problem, conditional_critic, verified_facts):
    debug = RecursionDebugHandler()
    try:
        result = problem_explainer_agent.with_config(callbacks=[debug, gemini_31_flash_lite_quota]).invoke(
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


def _next_problem_id(problem_tree):
    ids = []

    def walk(node):
        if not isinstance(node, dict):
            return
        if isinstance(node.get("problem_id"), int):
            ids.append(node["problem_id"])
        for action in node.get("action_plans", []):
            child = action.get("new_problem")
            if child:
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
            child = action.get("new_problem")
            if child:
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


def _attach_action_plans(
    current_problem,
    advisor_strategy,
    critic_results,
    explainer_results,
    runtime,
    auto_runtime,
):
    """Attach every action, its critic verdict, and explainer result to the problem node."""

    explain_by_critic = {item["critic_id"]: item for item in explainer_results}

    current_problem["action_plans"] = []

    for critic_result in critic_results:
        critic_id = critic_result["critic_id"]
        action_item = critic_result.get("action")
        critique = critic_result.get("critique") or {}

        action_node = {
            "action_plan": action_item,
            "critic_id": critic_id,
            "runtime": runtime,
            "auto_runtime": auto_runtime,
            "verdict": critique.get("verdict") if isinstance(critique, dict) else None,
            "verdict_reason": (
                critique.get("verdict_reason")
                if isinstance(critique, dict)
                else None
            ),
            "new_problem": None,
        }

        explainer = explain_by_critic.get(critic_id)
        if explainer is not None:
            result = explainer.get("result")
            if result is not None:
                action_node["new_problem"] = explainer.get("new_problem")
                action_node["new_problem_classification"] = result.classification
                action_node["new_problem_reason"] = result.reason
            else:
                action_node["new_problem_classification"] = None
                action_node["new_problem_reason"] = None

        current_problem["action_plans"].append(action_node)


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
            child = action.get("new_problem")
            if child:
                walk(child)

    for root in problem_tree:
        walk(root)

    return maximum


def run_workflow(
    history,
    verified_facts_memory=None,
    runtime=1,
    auto_runtime=0,
    problem_tree=None,
    critic_id_counter=0,
    active_critics=None,
    max_problem_depth=3,
    # Kept only for compatibility with older callers. It is not used for traversal.
    problem_stack=None,
    current_problem_index=0,
):
    """
    BFS orchestration around the existing agent pipeline.

    Existing agents remain intact:
        Problem Identifier -> Advisor -> Verified Facts -> Classifier
        -> Systematic Advice Converter -> ALL Critics -> ALL Problem Explainers
        -> queue discovered child problems -> Advisor serially, breadth-first.

    The critical invariant is that no child problem is sent to Advisor until
    every critic and every Problem Explainer for the current problem has finished.
    """

    if verified_facts_memory is None:
        verified_facts_memory = []
    if active_critics is None:
        active_critics = []

    # Migrate the old stack only as input compatibility; all new traversal is tree-based.
    if problem_tree is None:
        problem_tree = []
        if problem_stack:
            # Preserve the existing root/problem objects as much as possible.
            # Nested legacy stack nodes are flattened into roots only for migration.
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

    # ------------------------------------------------------------
    # BFS queue.
    # Root is processed first. Children discovered from one complete
    # level are appended after ALL explainers of that level finish.
    # ------------------------------------------------------------
    queue = deque()
    root = problem_tree[0]

    # If root already has a completed solution, this call is normally a continuation.
    queue.append((root, list(history)))

    last_result = None
    human_turn_result = None

    while queue:
        current_problem, current_history = queue.popleft()
        current_problem_id = current_problem["problem_id"]
        current_auto_runtime = current_problem["auto_runtime"]

        # Do not solve an already solved node again.
        if current_problem.get("status") == "SOLVED":
            continue

        # Verified Facts is extracted only when this workflow execution
        # was started by a real user turn. Child problems are automatic
        # Advisor calls and must reuse the existing verified memory.
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
        last_result = result
        if human_turn_result is None:
            human_turn_result = result

        all_logs.extend(result["logs"])
        all_critic_results.extend(result["critic_results"])
        all_critic_records.extend(result["critic_records"])
        critic_id_counter = result["critic_id_counter"]
        active_critics = result["active_critics"]

        if result["verified_facts"] is not None:
            current_verified_facts.append(result["verified_facts"].model_dump())

        if result["needs_input"]:
            current_problem["status"] = "WAITING_FOR_USER"
            return {
                **result,
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

        systematic_advice = result["systematic_advice"]
        if systematic_advice is None:
            current_problem["status"] = "ERROR"
            return {
                **result,
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

        # --------------------------------------------------------
        # IMPORTANT BFS BARRIER:
        # ALL N critics have already completed inside run_core_workflow.
        # Now run Problem Explainer for ALL N critic results before
        # creating or queueing ANY child problem.
        # --------------------------------------------------------
        explainer_results = []

        for critic_result in result["critic_results"]:
            if critic_result.get("critique") is None:
                explainer_results.append(
                    {
                        "critic_id": critic_result["critic_id"],
                        "result": None,
                        "debug": None,
                        "status": "SKIPPED_CRITIC_ERROR",
                        "new_problem": None,
                    }
                )
                continue

            explainer = explain_problem(
                current_problem=current_problem,
                conditional_critic=critic_result["critique"],
                verified_facts=current_verified_facts,
            )

            explainer_result = explainer["result"]
            child_payload = None

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

            if (
                explainer_result is not None
                and explainer_result.classification == "NEW_PROBLEM"
                and explainer_result.problem
            ):
                child_payload = explainer_result

            explainer_results.append(
                {
                    "critic_id": critic_result["critic_id"],
                    "result": explainer_result,
                    "debug": explainer["debug"],
                    "status": explainer["status"],
                    "new_problem": None,
                    "child_payload": child_payload,
                }
            )

        # --------------------------------------------------------
        # All explainers are complete. Only now can children be built.
        # --------------------------------------------------------
        _attach_action_plans(
            current_problem=current_problem,
            advisor_strategy=systematic_advice.model_dump(),
            critic_results=result["critic_results"],
            explainer_results=explainer_results,
            runtime=current_problem["runtime"],
            auto_runtime=current_problem["auto_runtime"],
        )

        next_children = []
        for action_node, explainer_item in zip(
            current_problem["action_plans"], explainer_results
        ):
            explainer_result = explainer_item.get("child_payload")
            if explainer_result is None:
                continue

            new_depth = current_problem.get("depth", 0) + 1
            if new_depth > max_problem_depth:
                action_node["new_problem"] = None
                action_node["new_problem_classification"] = "MAX_DEPTH_REACHED"
                action_node["new_problem_reason"] = (
                    "A new problem was identified, but the configured maximum problem depth was reached."
                )
                continue

            new_problem_id = _next_problem_id(problem_tree)
            next_auto_runtime += 1
            child_auto_runtime = next_auto_runtime

            child_node = _build_problem_node(
                problem_id=new_problem_id,
                parent_problem_id=current_problem_id,
                depth=new_depth,
                problem=explainer_result.problem,
                runtime=runtime,
                auto_runtime=child_auto_runtime,
                source_critic_id=explainer_item["critic_id"],
            )

            action_node["new_problem"] = child_node
            action_node["new_problem_classification"] = "NEW_PROBLEM"
            action_node["new_problem_reason"] = explainer_result.reason

            next_children.append(
                (
                    child_node,
                    _automatic_problem_history(
                        base_history=current_history,
                        child_problem=child_node,
                        source_critic=next(
                            (
                                c["critique"]
                                for c in result["critic_results"]
                                if c["critic_id"] == explainer_item["critic_id"]
                            ),
                            {},
                        ),
                        verified_facts=current_verified_facts,
                    ),
                )
            )
            all_child_problems.append(child_node)

        current_problem["solution"] = systematic_advice.model_dump()
        current_problem["status"] = (
            "WAITING_FOR_CHILDREN" if next_children else "SOLVED"
        )

        # Append all siblings only after ALL current-level explainers completed.
        for child in next_children:
            queue.append(child)

    # If BFS has exhausted, every queued problem has completed.
    # Mark ancestors with no unresolved children as solved.
    def finalize(node):
        unresolved = False
        for action in node.get("action_plans", []):
            child = action.get("new_problem")
            if child:
                finalize(child)
                if child.get("status") not in {"SOLVED", "ERROR", "WAITING_FOR_USER"}:
                    unresolved = True
        if node.get("status") == "WAITING_FOR_CHILDREN" and not unresolved:
            node["status"] = "SOLVED"

    for root_node in problem_tree:
        finalize(root_node)

    # The last result provides the normal UI response. Root response is used
    # for the human turn because child problem solving is automatic.
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
