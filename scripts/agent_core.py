"""
Headless agent core: the multi-part prompt→LLM→edit→execute loop
without any Streamlit dependency. All imports (LLM_methods,
code_edit_utils, document_utils, constants) are Streamlit-free.

Usage:
    from agent_core import AgentState, agent_loop

    state = AgentState(
        project_dir="playground/my_project",
        selected_files=["playground/my_project/main.py"],
        model_provider="Anthropic",
    )
    result = agent_loop(state, "Write a hello world script in scratch.py")
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import os

from dependencies import (
    # LLM_methods
    ConversationManager,
    generate_completion,
    # conversation_storage
    save_conversation,
    # edit_models
    Edit,
    CodeEdit,
    EntryEdit,
    EditIntent,
    EditReport,
    EditResult,
    should_continue,
    resolve_all,
    apply_all,
    format_edit_log,
    parse_edits,
    run_scratch_script,
    # document_utils
    generate_timestamp,
    # constants
    Base_Prompt,
    Code_Instructions,
    Scratch_Instructions,
)


# ---------------------------------------------------------------------------
# AgentState — replaces st.session_state for the headless loop
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """All state the agent loop needs, no Streamlit."""

    # Project / files
    project_dir: str = "playground/temp"
    selected_files: list = field(default_factory=list)
    uploaded_files: dict = field(default_factory=dict)  # name -> content

    # Model
    model_provider: str = "Anthropic"  # "Anthropic" | "OpenAI"
    api_key_type: str = "research"
    model_name: str | None = None  # Explicit model override, e.g. "claude-opus-4-5-20251101"

    # Mode: "edit" (full file editing) or "base" (scratch.py only)
    mode: str = "base"

    # Agent iteration limits
    max_parts: int = 3
    max_output_chars: int = 20000
    exec_timeout: int = 90
    max_history: int = 50

    # Entry context (optional — for document-tree-aware agents)
    graph: object = None  # NetworkX DiGraph or None
    current_entry_id: Optional[str] = None
    context_root_id: Optional[str] = None
    context_token_budget: int = 10000

    # Conversation state
    conversation: ConversationManager = field(default_factory=ConversationManager)

    # Callbacks for streaming output (default: print)
    on_assistant_chunk: object = None  # callable(str) -> None
    on_status: object = None  # callable(str) -> None

    # Token usage accumulator (populated by agent_loop)
    usage_log: list = field(default_factory=list)

    # Whether to persist conversation to disk after the loop
    save_conversation_to_disk: bool = True

    def __post_init__(self):
        if self.on_assistant_chunk is None:
            self.on_assistant_chunk = lambda chunk: print(chunk, end="", flush=True)
        if self.on_status is None:
            self.on_status = lambda msg: print(f"[status] {msg}")

    @property
    def total_usage(self) -> dict:
        """Aggregate token usage across all LLM calls in this session."""
        totals = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        for u in self.usage_log:
            for k in totals:
                totals[k] += u.get(k, 0)
        return totals

    @property
    def estimated_cost_usd(self) -> float:
        """Estimate cost using Opus pricing: $5/M input, $25/M output, $6.25/M cache write, $0.50/M cache read."""
        u = self.total_usage
        return (
            u["input_tokens"] * 5.0 / 1_000_000
            + u["output_tokens"] * 25.0 / 1_000_000
            + u["cache_creation_input_tokens"] * 6.25 / 1_000_000
            + u["cache_read_input_tokens"] * 0.50 / 1_000_000
        )


# ---------------------------------------------------------------------------
# Context assembly — pure functions replacing build_context_summary / prepare_message_context
# ---------------------------------------------------------------------------

def build_file_context(state: AgentState) -> str:
    """Build the file contents portion of the system prompt."""
    context = "\nCurrent files:\n"
    lang_map = {".py": "python", ".md": "markdown", ".tex": "latex", ".bib": "bibtex", ".yml": "yaml"}

    for file_path in state.selected_files:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                content = f.read()
            ext = os.path.splitext(file_path)[1]
            lang = lang_map.get(ext, "text")
            context += f"\n{file_path}:\n```{lang}\n{content}\n```\n"

    # Uploaded files
    if state.uploaded_files:
        context += "\nUploaded text files:\n"
        for name, content in state.uploaded_files.items():
            ext = os.path.splitext(name)[1]
            lang = "latex" if ext == ".tex" else ("markdown" if ext == ".md" else "json" if ext == ".json" else "text")
            context += f"\n{name}:\n```{lang}\n{content}\n```\n"

    return context


def build_entry_context(state: AgentState) -> str:
    """Build entry context from the document tree, if available.
    
    In standalone/headless mode, graph is always None so this returns "".
    """
    if state.graph is None or state.current_entry_id is None:
        return ""
    # Full implementation requires document_utils.create_context_partition
    # and format_entry_for_context — not needed for headless pre-reg.
    return ""


def build_context_summary_pure(state: AgentState, execution_output: str | None = None,
                               current_part: int | None = None, max_parts: int | None = None) -> dict:
    """
    Pure version of build_context_summary. Returns the same dict shape
    as the Streamlit version but reads from AgentState instead of session_state.
    """
    base_prompt_chars = len(Base_Prompt)

    # File sizes
    total_file_chars = 0
    file_count = 0
    for fp in state.selected_files:
        if os.path.exists(fp):
            total_file_chars += os.path.getsize(fp)
            file_count += 1

    # Entry context (always empty in headless mode)
    entry_context_chars = 0
    entry_count = 0
    entry_context_content = ""

    # Uploaded files
    uploaded_chars = sum(len(c) for c in state.uploaded_files.values())
    uploaded_count = len(state.uploaded_files)

    # History
    history = get_history_from_conversation(state.conversation, state.max_history)
    history_chars = sum(len(t.get("user") or "") + len(t.get("assistant") or "") for t in history)
    history_turns = len(history)

    # Total
    total_chars = base_prompt_chars + total_file_chars + entry_context_chars + uploaded_chars + history_chars
    total_tokens = total_chars // 4

    # Scope display
    scope = "global" if (state.context_root_id and state.context_root_id != state.current_entry_id) else "local"
    root_display = (state.context_root_id or state.current_entry_id or "N/A")
    if root_display and len(root_display) > 20:
        root_display = root_display[:20] + "..."

    ts = datetime.now().strftime("%A, %B %d, %Y %H:%M")

    # Files section
    files_section = ""
    if file_count > 0 or uploaded_count > 0:
        files_lines = ["\n**Files in Context:**"]
        if file_count > 0:
            file_names = [os.path.basename(f) for f in state.selected_files]
            files_lines.append(f"- Editable ({file_count}): {', '.join(file_names)}")
        if uploaded_count > 0:
            files_lines.append(f"- Uploaded ({uploaded_count}): {', '.join(state.uploaded_files.keys())}")
        files_section = "\n".join(files_lines)

    exec_section = ""
    if execution_output:
        exec_section = f"\n\n**Execution Output:**\n```\n{execution_output}\n```"

    summary_parts = [
        f"**Mode:** `{state.mode}` | **Scope:** `{scope}` | **Root:** `{root_display}` | **Project:** `{state.project_dir}` | **Time:** {ts}"
        + (f" | **Turn:** {current_part}/{max_parts}" if current_part is not None else ""),
        f"""
## Context Summary
**Total: ~{total_tokens:,} tokens**

| Component | ~Tokens |
|:----------|--------:|
| Base Prompt | {base_prompt_chars//4:,} |
| Code Files ({file_count}) ✅ | {total_file_chars//4:,} |
| Entries ({entry_count}) | {entry_context_chars//4:,} |
| Uploads ({uploaded_count}) | {uploaded_chars//4:,} |
| History ({history_turns} turns) | {history_chars//4:,} |""",
    ]

    if files_section:
        summary_parts.append(files_section)
    if exec_section:
        summary_parts.append(exec_section)

    summary_string = "\n".join(summary_parts)

    return {
        "mode": state.mode,
        "total_tokens": total_tokens,
        "summary_string": summary_string,
        "entry_context_content": entry_context_content,
    }


# ---------------------------------------------------------------------------
# Conversation history — pure version of get_conversation_history
# ---------------------------------------------------------------------------

def get_history_from_conversation(cm: ConversationManager, max_turns: int = 50) -> list[dict]:
    """Walk the conversation tree and return history dicts."""
    history = []
    node = cm.root
    for index in cm.current_path:
        node = node["children"][index]
        user_msg = node.get("user")
        assistant_msg = node.get("assistant")
        if user_msg is None and assistant_msg is None:
            continue
        history.append({"user": user_msg, "assistant": assistant_msg})

    if max_turns and len(history) > max_turns:
        history = history[-max_turns:]
    return history


# ---------------------------------------------------------------------------
# The core agent loop
# ---------------------------------------------------------------------------

def format_execution_as_user_message(output: str) -> str:
    output = "" if output is None else str(output)
    return "[Execution completed]\n```output\n" + output.rstrip() + "\n```"


def _add_turn(cm: ConversationManager, user_msg, assistant_msg, *, turn_state=None):
    """Add a turn and advance current_path into the new child."""
    cm.add_turn(user_msg, assistant_msg, [], turn_state=turn_state)
    # add_turn() always appends to current_path, so no fallback needed.


def agent_loop(state: AgentState, prompt: str) -> dict:
    """
    Run the multi-part agent loop headlessly.

    Returns a dict with:
        - "parts": list of {"role": str, "content": str} for each message produced
        - "edit_results": list of edit result dicts from last part (or None)
        - "final_output": last execution output (or "")
        - "turn_count": number of LLM calls made
    """
    cm = state.conversation
    turn_timestamp = generate_timestamp()

    # Choose system prompt skeleton based on mode
    scratch_only = (state.mode == "base")
    instructions = Scratch_Instructions if scratch_only else Code_Instructions

    parts = []  # All messages produced in this interaction
    current_part = 0
    continuation_prompt = prompt
    last_edit_results = None
    last_exec_output = ""

    while current_part < state.max_parts:
        current_part += 1

        # Build system prompt with current file contents (may change between parts)
        file_context = build_file_context(state)
        entry_context = build_entry_context(state)
        system_prompt = Base_Prompt + instructions + file_context + entry_context

        # Build turn context header for the LLM
        ctx = build_context_summary_pure(state, current_part=current_part, max_parts=state.max_parts)
        llm_prompt = f"```system_message\n{ctx['summary_string']}\n```\n\n{continuation_prompt}"

        # Get conversation history
        history = get_history_from_conversation(cm, state.max_history)

        # Stream LLM response
        state.on_status(f"Part {current_part}/{state.max_parts}: calling LLM...")
        part_response = ""
        part_usage = {}
        def _capture_usage(u):
            nonlocal part_usage
            part_usage = u

        for chunk in generate_completion(
            llm_prompt,
            system_prompt,
            history,
            provider=state.model_provider,
            api_key_type=state.api_key_type,
            model_name=state.model_name,
            usage_callback=_capture_usage,
        ):
            part_response += chunk
            state.on_assistant_chunk(chunk)

        if part_usage:
            state.usage_log.append(part_usage)

        # Newline after streaming
        state.on_assistant_chunk("\n")
        parts.append({"role": "assistant", "content": part_response})

        # Parse and apply edits via abstract Edit objects
        edit_log = None
        exec_output = ""

        edits = parse_edits(
            part_response,
            mode="base" if scratch_only else "edit",
            context_files=state.selected_files if not scratch_only else [],
        )

        # Resolve all edits (handles scratch_only, path resolution, allowlists)
        reports = None
        if edits:
            resolve_all(
                edits,
                context_files=state.selected_files if not scratch_only else [],
                project_dir=state.project_dir,
                scratch_only=scratch_only,
            )

            # Apply all resolved edits
            reports = apply_all(edits)
            edit_log = format_edit_log(reports)
            last_edit_results = reports

            # Add newly created files to selected_files (edit mode only)
            if not scratch_only:
                for edit, report in zip(edits, reports):
                    if report.created and isinstance(edit, CodeEdit) and not edit.is_scratch:
                        new_path = os.path.abspath(edit.resolved_path)
                        existing = [os.path.abspath(f) for f in state.selected_files]
                        if new_path not in existing:
                            state.selected_files.append(new_path)

            # Execute scratch.py if any scratch edit succeeded
            scratch_succeeded = any(
                isinstance(e, CodeEdit) and e.is_scratch and r.success
                for e, r in zip(edits, reports)
            )
            if scratch_succeeded:
                state.on_status("Executing scratch.py...")
                exec_output = run_scratch_script(
                    state.project_dir,
                    timeout=state.exec_timeout,
                    max_chars=state.max_output_chars,
                )
                last_exec_output = exec_output

                # In base mode, discover new files created by scratch.py
                # so the agent sees its own outputs on subsequent turns
                if scratch_only:
                    existing_abs = set(os.path.abspath(f) for f in state.selected_files)
                    # Only auto-add code/markdown files — data accessed via scratch.py
                    context_extensions = {'.py', '.md', '.txt', '.yml', '.yaml'}
                    for fname in sorted(os.listdir(state.project_dir)):
                        fpath = os.path.join(state.project_dir, fname)
                        if (os.path.isfile(fpath)
                                and os.path.abspath(fpath) not in existing_abs
                                and not fname.startswith('.')
                                and not fname.startswith('__')):
                            ext = os.path.splitext(fname)[1].lower()
                            if ext not in context_extensions:
                                continue
                            state.selected_files.append(os.path.abspath(fpath))
                            state.on_status(f"New file in context: {fname}")

                if current_part >= state.max_parts:
                    indented = "\n".join("    " + line for line in exec_output.split("\n"))
                    edit_log = (edit_log or "").rstrip() + f"\n\n## Execution Output (final part):\n{indented}"
                else:
                    edit_log = (edit_log or "").rstrip() + (
                        "\n\n## Scratch Execution\nExecution completed; output stored as an auto-generated user message."
                    )

        # Persist turn
        turn_state = {"timestamp": turn_timestamp if current_part == 1 else generate_timestamp()}
        if edit_log:
            turn_state["logs"] = {"edits": edit_log}

        # Persist this part as a complete (user, assistant) node.
        # Part 1 uses the original user prompt; continuations use the auto-prompt.
        user_for_node = prompt if current_part == 1 else continuation_prompt
        _add_turn(cm, user_for_node, part_response, turn_state=turn_state)

        # Check for auto-continuation
        if reports is None:
            break

        reason_type, reason_detail = should_continue(edits, reports, exec_output)

        if reason_type and current_part < state.max_parts:
            if exec_output:
                auto_msg = format_execution_as_user_message(exec_output)
                if reason_type != "continuing":
                    auto_msg += f"\n\nThe previous attempt had an issue: {reason_detail}. Please fix and try again."
            else:
                auto_msg = (
                    f"The previous attempt had an issue: {reason_detail}. Please fix and try again."
                    if reason_type != "continuing"
                    else "Continue."
                )

            parts.append({"role": "user", "content": auto_msg})
            # Don't create a separate node for the auto-prompt.
            # Store it as the user message for the next iteration,
            # which will commit (auto_msg, next_response) as a single node.
            continuation_prompt = auto_msg
            state.on_status(f"Auto-continuing: {reason_detail}")
            continue

        break

    # Persist conversation to disk
    if state.save_conversation_to_disk:
        try:
            save_conversation(cm)
        except Exception as e:
            state.on_status(f"Warning: could not save conversation: {e}")

    return {
        "parts": parts,
        "edit_results": last_edit_results,
        "final_output": last_exec_output,
        "turn_count": current_part,
        "usage": state.total_usage,
        "estimated_cost_usd": state.estimated_cost_usd,
        "conversation_id": cm.conversation_id,
    }