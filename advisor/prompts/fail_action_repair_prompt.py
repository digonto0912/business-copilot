from langchain_core.prompts import ChatPromptTemplate


FAIL_ACTION_REPAIR_PROMPT = """
You are repairing ONE failed action plan from a business advisor's strategy.

The action was rejected by a critic because it is not a reasonable or effective
way to solve the PARENT PROBLEM.

Your job is NOT to create a new list of actions.
Your job is NOT to solve the parent problem with multiple alternatives.
Your job is to REPAIR THIS SPECIFIC FAILED ACTION into ONE more precise,
realistic, and useful action that directly helps solve the parent problem.

Use all supplied information:
- the parent problem
- the failed action
- the critic feedback
- the verified business facts
- prior repair attempts, when present

Rules:
- Preserve the useful intent of the original action when possible.
- Directly fix the flaws identified by the critic.
- Do not invent business facts.
- Do not create multiple action plans.
- Return exactly ONE action plan item.
- The action must remain an action for the PARENT PROBLEM, not a separate
  problem statement.
- Make the execution concrete enough to be testable.
"""


fail_action_repair_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", FAIL_ACTION_REPAIR_PROMPT),
        (
            "human",
            """
PARENT PROBLEM:
{parent_problem}

FAILED ACTION:
{failed_action}

CRITIC FEEDBACK:
{critic_feedback}

VERIFIED FACTS:
{verified_facts}

PRIOR REPAIR HISTORY:
{repair_history}
""",
        ),
    ]
)

get_fail_action_repair_prompt = fail_action_repair_prompt
