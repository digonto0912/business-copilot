import json

import streamlit as st
import dotenv

from recursion_workflow import run_workflow


dotenv.load_dotenv()


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Business Advisor",
    page_icon="💼",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    html,
    body,
    [data-testid="stAppViewContainer"],
    [data-testid="stApp"] {
    }


    [data-testid="stAppViewContainer"] > .main {
    }


    .block-container {

        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;

    }


    h1 {

        margin-top: 0 !important;
        margin-bottom: 0.2rem !important;
    }


    [data-testid="stHorizontalBlock"] {

        min-height: 0 !important;

        align-items: stretch !important;
    }


    [data-testid="column"] {

        height: 100% !important;

        min-height: 0 !important;

        display: flex !important;
        flex-direction: column !important;
    }


    [data-testid="column"] > div {

        height: 100% !important;

        min-height: 0 !important;

        max-height: 100% !important;
    }


    [data-testid="column"] [data-testid="stVerticalBlock"] {

        min-height: 0 !important;
    }


    [data-testid="stVerticalBlockBorderWrapper"] {

        min-height: 0 !important;

        max-height: 100% !important;
    }


    [data-testid="stVerticalBlock"] {

        gap: 0.35rem !important;
    }


    textarea {

        resize: none !important;
    }


    [data-testid="stMarkdownContainer"] p {

        margin-bottom: 0.25rem;
    }


    button {

        margin-top: 0 !important;
    }


    [data-testid="stCode"] {

        max-width: 100% !important;

    }


    [data-testid="stCode"] pre {

        white-space: pre-wrap !important;

        word-break: break-word !important;

        max-width: 100% !important;
    }


    [data-testid="stCode"] code {

        white-space: pre-wrap !important;

        word-break: break-word !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


if "needs_input" not in st.session_state:
    st.session_state.needs_input = True


if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []


if "verified_facts" not in st.session_state:
    st.session_state.verified_facts = []


if "critic_results" not in st.session_state:
    st.session_state.critic_results = []


# ============================================================
# MEMORY / EXECUTION STATE
# ============================================================

if "runtime" not in st.session_state:
    st.session_state.runtime = 0


if "auto_runtime" not in st.session_state:
    st.session_state.auto_runtime = 0


if "problem_tree" not in st.session_state:
    st.session_state.problem_tree = []


if "critic_id_counter" not in st.session_state:
    st.session_state.critic_id_counter = 0


if "critic_records" not in st.session_state:
    st.session_state.critic_records = []


if "active_critics" not in st.session_state:
    st.session_state.active_critics = []


if "max_problem_depth" not in st.session_state:
    st.session_state.max_problem_depth = 3


# ============================================================
# TITLE
# ============================================================

st.title("💼 Business Advisor")


# ============================================================
# TWO COLUMNS
# ============================================================

left, right = st.columns(
    [2, 1],
    gap="large",
)


# ============================================================
# LEFT COLUMN
# ============================================================

with left:
    st.subheader("💬 Chat")

    # --------------------------------------------------------
    # PROBLEM DEPTH CONTROL
    # --------------------------------------------------------

    st.number_input(
        "Maximum Problem Depth",
        min_value=0,
        max_value=20,
        value=st.session_state.max_problem_depth,
        step=1,
        key="max_problem_depth",
        help=("Maximum recursive depth allowed for child problems."),
    )

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    chat_box = st.container(
        height=520,
        border=True,
    )

    with chat_box:
        if not st.session_state.messages:
            st.info("Tell me about your business to begin.")

        else:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])

                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    if st.session_state.needs_input:
        user_input = st.text_area(
            "Message",
            placeholder="Tell me about your business...",
            height=65,
            label_visibility="collapsed",
        )

        send = st.button(
            "➤ Send",
            type="primary",
            use_container_width=True,
        )

        # ----------------------------------------------------
        # PROCESS
        # ----------------------------------------------------

        if send:
            if not user_input.strip():
                st.warning("Please enter a message.")

            else:
                user_input = user_input.strip()

                # --------------------------------------------
                # USER STARTS A NEW RUNTIME
                # --------------------------------------------

                st.session_state.runtime += 1

                # A human-started flow always starts
                # with auto_runtime = 0.

                st.session_state.auto_runtime = 0

                # --------------------------------------------
                # SAVE USER MESSAGE
                # --------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": user_input,
                    }
                )

                # --------------------------------------------
                # BUILD HISTORY
                # --------------------------------------------

                history = []

                for message in st.session_state.messages:
                    history.append(
                        (
                            message["role"],
                            message["content"],
                        )
                    )

                # --------------------------------------------
                # RUN WORKFLOW
                # --------------------------------------------

                result = run_workflow(
                    history,
                    verified_facts_memory=(st.session_state.verified_facts),
                    runtime=(st.session_state.runtime),
                    auto_runtime=(st.session_state.auto_runtime),
                    problem_tree=(st.session_state.problem_tree),
                    critic_id_counter=(st.session_state.critic_id_counter),
                    active_critics=(st.session_state.active_critics),
                    max_problem_depth=(st.session_state.max_problem_depth),
                )

                # --------------------------------------------
                # SAVE ASSISTANT RESPONSE
                # --------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["response"],
                    }
                )

                # --------------------------------------------
                # SAVE AGENT LOGS
                # --------------------------------------------

                st.session_state.agent_logs.extend(result["logs"])

                # --------------------------------------------
                # SAVE VERIFIED FACTS
                # --------------------------------------------

                if result["verified_facts"] is not None:
                    st.session_state.verified_facts.append(
                        result["verified_facts"].model_dump()
                    )

                # --------------------------------------------
                # SAVE CRITIC RESULTS
                # --------------------------------------------

                if result["critic_results"]:
                    st.session_state.critic_results.extend(result["critic_results"])

                # --------------------------------------------
                # SAVE CRITIC RECORDS
                # --------------------------------------------

                if result["critic_records"]:
                    st.session_state.critic_records.extend(result["critic_records"])

                # --------------------------------------------
                # SAVE ACTIVE CRITICS
                # --------------------------------------------

                st.session_state.active_critics = result["active_critics"]

                # --------------------------------------------
                # SAVE PROBLEM TREE
                # --------------------------------------------

                st.session_state.problem_tree = result["problem_tree"]

                # --------------------------------------------
                # SAVE CRITIC ID COUNTER
                # --------------------------------------------

                st.session_state.critic_id_counter = result["critic_id_counter"]

                # --------------------------------------------
                # UPDATE INPUT STATE
                # --------------------------------------------

                st.session_state.needs_input = result["needs_input"]

                # --------------------------------------------
                # RERUN
                # --------------------------------------------

                st.rerun()

    else:
        st.success("The advisor has enough information.")


# ============================================================
# RIGHT COLUMN
# ============================================================

with right:
    st.subheader("🔍 Agent Debug")

    st.caption("Actual LLM input → raw LLM output")

    # --------------------------------------------------------
    # DEBUG HISTORY
    # --------------------------------------------------------

    debug_box = st.container(
        height=620,
        border=True,
    )

    with debug_box:
        if not st.session_state.agent_logs:
            st.info("No agent activity yet.")

        else:
            for index, log in enumerate(st.session_state.agent_logs):
                icon = "🟢" if log["status"] == "SUCCESS" else "🔴"

                latest = index == len(st.session_state.agent_logs) - 1

                with st.expander(
                    (
                        f"{icon} "
                        f"runtime={log.get('runtime')} "
                        f"auto={log.get('auto_runtime')} "
                        f"problem={log.get('problem_id')} "
                        f"| {log['agent']}"
                    ),
                    expanded=latest,
                ):
                    # ----------------------------------------
                    # PROMPT
                    # ----------------------------------------

                    st.markdown("### 📥 Prompt sent to LLM")

                    if isinstance(
                        log["prompt"],
                        list,
                    ):
                        for message in log["prompt"]:
                            role = message.get(
                                "role",
                                "unknown",
                            )

                            content = message.get(
                                "content",
                                "",
                            )

                            st.markdown(f"**{role.upper()}**")

                            st.code(
                                content,
                                language="text",
                            )

                    elif isinstance(
                        log["prompt"],
                        dict,
                    ):
                        st.code(
                            json.dumps(
                                log["prompt"],
                                indent=2,
                                ensure_ascii=False,
                            ),
                            language="json",
                        )

                    else:
                        st.code(
                            str(log["prompt"]),
                            language="text",
                        )

                    # ----------------------------------------
                    # RAW OUTPUT
                    # ----------------------------------------

                    st.markdown("### 📤 Raw LLM Response")

                    if isinstance(
                        log["output"],
                        (dict, list),
                    ):
                        st.code(
                            json.dumps(
                                log["output"],
                                indent=2,
                                ensure_ascii=False,
                            ),
                            language="json",
                        )

                    else:
                        st.code(
                            str(log["output"]),
                            language="text",
                        )

                    # ----------------------------------------
                    # STATUS
                    # ----------------------------------------

                    st.write(f"Status: {log['status']}")


    # ========================================================
    # DOWNLOAD FULL SESSION
    # ========================================================

    st.download_button("⬇️ Download Full Session", json.dumps({k: v for k, v in st.session_state.items()}, indent=2, ensure_ascii=False, default=str), "full_session.json", "application/json", use_container_width=True)


    # ========================================================
    # VERIFIED FACTS
    # ========================================================

    if st.button(
        "🧠 Show Verified Facts",
        use_container_width=True,
    ):
        with st.expander(
            "Verified Facts — Raw JSON",
            expanded=True,
        ):
            st.code(
                json.dumps(
                    st.session_state.verified_facts,
                    indent=2,
                    ensure_ascii=False,
                ),
                language="json",
            )

    # ========================================================
    # CRITIC RESULTS
    # ========================================================

    if st.button(
        "🔍 Show Critic Results",
        use_container_width=True,
    ):
        with st.expander(
            "Critic Results — Raw JSON",
            expanded=True,
        ):
            st.code(
                json.dumps(
                    st.session_state.critic_results,
                    indent=2,
                    ensure_ascii=False,
                ),
                language="json",
            )

    # ========================================================
    # PROBLEM STACK
    # ========================================================

    if st.button(
        "🌳 Show Problem Tree",
        use_container_width=True,
    ):
        with st.expander(
            "Problem Tree — Raw JSON",
            expanded=True,
        ):
            st.code(
                json.dumps(
                    st.session_state.problem_tree,
                    indent=2,
                    ensure_ascii=False,
                ),
                language="json",
            )

    # ========================================================
    # CRITIC RECORDS
    # ========================================================

    if st.button(
        "🧾 Show Critic Records",
        use_container_width=True,
    ):
        with st.expander(
            "Critic Records — Raw JSON",
            expanded=True,
        ):
            st.code(
                json.dumps(
                    st.session_state.critic_records,
                    indent=2,
                    ensure_ascii=False,
                ),
                language="json",
            )

    # ========================================================
    # ACTIVE CRITICS
    # ========================================================

    if st.button(
        "⚠️ Show Active Critics",
        use_container_width=True,
    ):
        with st.expander(
            "Active Critics — Raw JSON",
            expanded=True,
        ):
            st.code(
                json.dumps(
                    st.session_state.active_critics,
                    indent=2,
                    ensure_ascii=False,
                ),
                language="json",
            )

    # ========================================================
    # EXECUTION STATE
    # ========================================================

    if st.button(
        "📊 Show Execution State",
        use_container_width=True,
    ):
        with st.expander(
            "Execution State — Raw JSON",
            expanded=True,
        ):
            st.code(
                json.dumps(
                    {
                        "runtime": st.session_state.runtime,
                        "auto_runtime": st.session_state.auto_runtime,
                        "max_problem_depth": st.session_state.max_problem_depth,
                        "critic_id_counter": st.session_state.critic_id_counter,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                language="json",
            )

    # ========================================================
    # CLEAR EVERYTHING
    # ========================================================

    if st.button(
        "🗑️ Clear All Memory",
        use_container_width=True,
    ):
        st.session_state.messages = []

        st.session_state.needs_input = True

        st.session_state.agent_logs = []

        st.session_state.verified_facts = []

        st.session_state.critic_results = []

        st.session_state.runtime = 0

        st.session_state.auto_runtime = 0

        st.session_state.problem_tree = []

        st.session_state.critic_id_counter = 0

        st.session_state.critic_records = []

        st.session_state.active_critics = []

        st.rerun()
