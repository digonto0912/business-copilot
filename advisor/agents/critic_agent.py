import json
from typing import Any, Callable, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, tool

from llm import llm

from prompts import get_critic_prompt

from schemas.critic_schema import ActionCritique


# ============================================================
# WEB SEARCH TOOL
# ============================================================

def _default_search_tool() -> BaseTool:

    @tool
    def web_search(query: str) -> str:
        """
        Search the web for a specific factual claim about the outside world.

        Use this only for CHECKABLE assumptions.
        Do not use it to discover facts about the business being evaluated.
        """

        try:

            from duckduckgo_search import DDGS

        except ImportError as exc:

            return (
                "web_search unavailable because "
                "duckduckgo_search is not installed: "
                f"{exc}"
            )

        try:

            with DDGS() as ddgs:

                results = list(
                    ddgs.text(
                        query,
                        max_results=5,
                    )
                )

        except Exception as exc:

            return (
                f"web_search failed for query '{query}': {exc}"
            )

        if not results:

            return (
                f"No results found for '{query}'."
            )

        lines = []

        for result in results:

            title = result.get(
                "title",
                "",
            )

            body = result.get(
                "body",
                "",
            )

            href = result.get(
                "href",
                "",
            )

            lines.append(
                f"- {title}: {body} (source: {href})"
            )

        return "\n".join(lines)

    return web_search


# ============================================================
# AGENT
# ============================================================

class CriticAgent:

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        search_tool: Optional[BaseTool] = None,
        max_tool_iterations: int = 5,
    ) -> None:

        self.llm = llm or globals()["llm"]

        self.search_tool = (
            search_tool
            or _default_search_tool()
        )

        self.max_tool_iterations = (
            max_tool_iterations
        )

        self._tools_by_name = {
            self.search_tool.name:
                self.search_tool
        }

        self._llm_with_tools = (
            self.llm.bind_tools(
                [
                    self.search_tool,
                    ActionCritique,
                ]
            )
        )

    # ========================================================
    # ONE ACTION
    # ========================================================

    def critique_action(
        self,
        action_item: dict[str, Any],
        verified_context: Any,
        advisor_strategy: dict[str, Any],
    ) -> ActionCritique:
        """
        Evaluate exactly ONE target action.

        The model receives:

        1. Original critic system instructions
        2. Verified business context
        3. Full advisor strategy context
        4. One target action

        Only the target action is judged.
        """

        # ----------------------------------------------------
        # Build the original system prompt
        # ----------------------------------------------------

        prompt = get_critic_prompt.invoke(
            {
                "verified_context": json.dumps(
                    verified_context,
                    indent=2,
                    ensure_ascii=False,
                ),
                "action_item": json.dumps(
                    action_item,
                    indent=2,
                    ensure_ascii=False,
                ),
            }
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Your prompt only contains a SYSTEM message.
        # Add the actual runtime data as a HUMAN message.
        #
        # ----------------------------------------------------

        human_content = (
            "ADVISOR STRATEGY CONTEXT\n"
            "=========================\n"
            "The following is the advisor's strategy reasoning "
            "and surrounding strategic context.\n\n"
            f"{json.dumps(advisor_strategy, indent=2, ensure_ascii=False)}\n\n"
            "IMPORTANT:\n"
            "Use this strategy context to understand the meaning, "
            "reasoning, mechanism, and intended role of the target "
            "action. Do not critique the other actions because they "
            "are not provided.\n\n"
            "Now evaluate the TARGET ACTION from the system prompt "
            "using the required five-stage process."
        )

        messages: list[BaseMessage] = [
            *prompt.to_messages(),
            HumanMessage(
                content=human_content
            ),
        ]

        return self._run_tool_loop(
            messages
        )

    # ========================================================
    # WHOLE PLAN
    # ========================================================

    def critique_plan(
        self,
        advisor_strategy: dict[str, Any],
        verified_context: Any,
    ) -> list[ActionCritique]:
        """
        Critique every action independently.

        Each call receives:
        - same verified context
        - same advisor strategy context
        - one target action
        """

        action_items = advisor_strategy.get(
            "prioritized_action_plan",
            [],
        )

        return [
            self.critique_action(
                action_item=item,
                verified_context=verified_context,
                advisor_strategy=advisor_strategy,
            )
            for item in action_items
        ]

    # ========================================================
    # STREAMING RESULTS
    # ========================================================

    def critique_plan_streaming(
        self,
        advisor_strategy: dict[str, Any],
        verified_context: Any,
        on_result: Callable[
            [dict[str, Any], ActionCritique],
            None,
        ],
    ) -> None:

        action_items = advisor_strategy.get(
            "prioritized_action_plan",
            [],
        )

        for item in action_items:

            critique = self.critique_action(
                action_item=item,
                verified_context=verified_context,
                advisor_strategy=advisor_strategy,
            )

            on_result(
                item,
                critique,
            )

    # ========================================================
    # TOOL LOOP
    # ========================================================

    def _run_tool_loop(
        self,
        messages: list[BaseMessage],
    ) -> ActionCritique:

        for _ in range(
            self.max_tool_iterations
        ):

            ai_message: AIMessage = (
                self._llm_with_tools.invoke(
                    messages
                )
            )

            messages.append(
                ai_message
            )

            tool_calls = (
                ai_message.tool_calls
                or []
            )

            # ------------------------------------------------
            # MODEL DID NOT CALL A TOOL
            # ------------------------------------------------

            if not tool_calls:

                messages.append(
                    HumanMessage(
                        content=(
                            "You must submit the final result "
                            "by calling ActionCritique exactly once. "
                            "Do not respond with free text."
                        )
                    )
                )

                continue

            final_result = None

            # ------------------------------------------------
            # PROCESS TOOL CALLS
            # ------------------------------------------------

            for call in tool_calls:

                name = call["name"]

                args = call["args"]

                call_id = call["id"]

                # --------------------------------------------
                # FINAL CRITIQUE
                # --------------------------------------------

                if name == "ActionCritique":

                    final_result = (
                        ActionCritique(
                            **args
                        )
                    )

                    messages.append(
                        ToolMessage(
                            content=(
                                "ActionCritique accepted."
                            ),
                            tool_call_id=call_id,
                        )
                    )

                # --------------------------------------------
                # WEB SEARCH
                # --------------------------------------------

                elif (
                    name
                    in self._tools_by_name
                ):

                    tool_output = (
                        self._tools_by_name[
                            name
                        ].invoke(
                            args
                        )
                    )

                    messages.append(
                        ToolMessage(
                            content=str(
                                tool_output
                            ),
                            tool_call_id=call_id,
                        )
                    )

                # --------------------------------------------
                # UNKNOWN TOOL
                # --------------------------------------------

                else:

                    messages.append(
                        ToolMessage(
                            content=(
                                f"Unknown tool '{name}'."
                            ),
                            tool_call_id=call_id,
                        )
                    )

            # ------------------------------------------------
            # FINAL RESULT
            # ------------------------------------------------

            if final_result is not None:

                return final_result

        raise RuntimeError(
            "Critic did not submit an ActionCritique "
            f"within {self.max_tool_iterations} "
            "tool-calling iterations."
        )


# ============================================================
# ATTACH CRITIQUES
# ============================================================

def attach_critiques_to_plan(
    plan: dict[str, Any],
    critiques: list[ActionCritique],
) -> dict[str, Any]:

    out = json.loads(
        json.dumps(plan)
    )

    for item, critique in zip(
        out.get(
            "prioritized_action_plan",
            [],
        ),
        critiques,
    ):

        item["critique"] = (
            critique.model_dump()
        )

    return out


# ============================================================
# HUMAN-READABLE REPORT
# ============================================================

def format_critique_report(
    critique: ActionCritique,
) -> str:

    lines = [
        f"=== {critique.action_title} ===",
        f"CLAIM: {critique.claim}",
        f"MECHANISM: {critique.mechanism}",
        "",
        "LOAD-BEARING ASSUMPTIONS:",
    ]

    for assumption in (
        critique.load_bearing_assumptions
    ):

        lines.append(
            f"  [{assumption.classification}] "
            f"{assumption.assumption}"
        )

        lines.append(
            f"      -> {assumption.evidence}"
        )

    if critique.structural_flags:

        lines.append("")
        lines.append(
            "STRUCTURAL FLAGS:"
        )

        for flag in (
            critique.structural_flags
        ):

            lines.append(
                f"  [{flag.flaw_type}] "
                f"{flag.explanation}"
            )

    if (
        critique.settled_as_fact_violation
    ):

        lines.append("")
        lines.append(
            "STATED-AS-FACT VIOLATION: "
            f"{critique.settled_as_fact_violation}"
        )

    lines.append("")
    lines.append(
        f"VERDICT: {critique.verdict}"
    )

    lines.append(
        f"REASON: {critique.verdict_reason}"
    )

    return "\n".join(lines)