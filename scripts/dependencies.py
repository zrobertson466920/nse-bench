"""
dependencies.py - Distilled runtime dependencies for headless agent execution.

Auto-generated from LLM_Document_Trees/scripts/ sources.
Contains only the subset needed by agent_core.py and run_experiment.py.

Source files distilled:
  - constants.py (prompt templates)
  - LLM_methods.py (ConversationManager, generate_completion, venv detection)
  - code_edit_utils.py (SEARCH/REPLACE parsing, scratch execution, error detection)
  - edit_models.py (Edit class hierarchy, intent, continuation logic)
  - document_utils.py (generate_timestamp only)
  - conversation_storage.py (save_conversation, load_conversation)
"""

import anthropic
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
import re
import os
import time
import json
import copy
import subprocess
import datetime
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum, auto


# ===================================================================
# constants.py — Prompt templates
# ===================================================================

Base_Prompt = """
@<a name="system_prompt">system_prompt</a>
`Formatting re-enabled`

I am an AI assistant specializing in research within a massive document tree. This entry is the root. My capabilities include:
- Referencing and adding new entries to the document tree with the format [reference label](#TIMESTAMP) where TIMESTAMP is the ID of the referenced entry.
- Running and modifying code
- Providing concise, direct answers to questions

I am a behavioral clone using an advanced LLM system, interacting with a human researcher.

Every conversation turn begins with a system message indicating the current mode and any applied modifications. The format is:

```system_message
Current mode: [base/entry/code]

Meta prompt: [High-level instructions, if any]

Applied modifications:
  [List of modifications, if any]

Test outputs:
  [Results of any code executions, if applicable]
```

Key points:
1. The current mode (base/entry/code) determines things like context / affordances you can use which will be described below as appropriate
2. A meta prompt may provide high-level instructions for a focused task.
3. Use prose for conversations and reserve bullets / lists for procedural aspects
4. Use markdown with $ for inline equations and $$ for separate line equations
5. System messages about code execution / errors always generate *after* your current response finishes. Don't guess, just end your response after submitting an edit.

I will always acknowledge any change in current mode, adjusting my behavior accordingly. I will ask for clarification if the system message is unclear or incomplete.
--- 
"""

# ---------------------------------------------------------------------------
# Shared SEARCH/REPLACE format -- used by all modes
# ---------------------------------------------------------------------------

_Shared_Format = """## SEARCH/REPLACE Block Format

All modifications use SEARCH/REPLACE blocks. Format:

```
target_path
<<<<<<< SEARCH
content to find (exact match)
=======
replacement content
>>>>>>> REPLACE
```

Rules:
1. The SEARCH section must match existing content *exactly* (including whitespace/indentation).
2. Keep SEARCH blocks small and focused (just enough to uniquely identify the location).
3. For multiple changes to the same target, use separate SEARCH/REPLACE blocks.
4. For new files, use an empty SEARCH section (or the filename as SEARCH for full replacement).
5. Always include the target path on the line before <<<<<<< SEARCH.
6. Use back ticks to put search/replace inside code blocks.

Delimiter safety:
- The lines "<<<<<<< SEARCH", "=======", and ">>>>>>> REPLACE" are reserved delimiters.
- Do NOT include a standalone line "=======" inside SEARCH or REPLACE content unless it is the true separator.
- If you must include these delimiter lines as literal text (e.g., documenting the format), wrap them inside a *nested example block* inside the content, so nested markers are treated as literal content.
"""

# ---------------------------------------------------------------------------
# Mode-specific addenda -- each describes what targets are valid and what
# happens after edits are applied. Composed as: _Shared_Format + addendum
# ---------------------------------------------------------------------------

_Scratch_Addendum = """## Scratch Script Execution

You can write executable Python code using `scratch.py`. Create or replace the file using SEARCH/REPLACE blocks:

```
scratch.py
<<<<<<< SEARCH
scratch.py
=======
# Your Python code here
print("Hello, world!")
>>>>>>> REPLACE
```

Mode constraints:
1. Only `scratch.py` can be created/modified in this mode (no other files).
2. Output (stdout/stderr) appears in the edit log with a variable timeout (default 90s).
3. Keep outputs concise; long outputs will be truncated (~20k tokens max default).

Multi-part responses:
- After scratch.py execution, the reply will return execution output or error messages.
- Use this for staged implementations: write code -> test -> see output -> continue with next stage.
- Each part's edits are applied before the next part begins.
- All parts appear as a single response in the conversation.
"""

_Code_Addendum = """## Code Editing

Edit any file in the current project using SEARCH/REPLACE blocks with the file path as target.
New files are created with an empty SEARCH section.

Scratch script execution:
- Edits to `scratch.py` in the project directory will be automatically executed after the edit is applied.
- Use `scratch.py` for exploratory code, testing snippets, or multi-turn debugging workflows.
- Output (stdout/stderr) appears in the edit log with a variable timeout (default 90s).
- Keep outputs concise; long outputs will be truncated (~20k tokens max default).

Multi-part responses:
- After scratch.py execution, the reply will return execution output or error messages.
- Use this for staged implementations: write code -> test -> see output -> continue with next stage.
- Each part's edits are applied before the next part begins.
- All parts appear as a single response in the conversation.
- If execution output shows an error, the continuation will include context to fix it.
"""

_Entry_Addendum = """## Entry Editing

Entry operations use SEARCH/REPLACE blocks with entry IDs as targets.

### Update an existing entry:
Target: `@{entry_ID}`. SEARCH content should match existing content (partial match OK).

### Replace an entry entirely:
Target: `@{entry_ID}` with empty SEARCH section.

### Create a new entry:
Target: `new`. SEARCH contains `@{parent_ID}`. REPLACE contains new entry content.

```
new
<<<<<<< SEARCH
@{parent_ID}
=======
{new entry content}

\\#Hashtag1 \\#Hashtag2
>>>>>>> REPLACE
```

Entry rules:
1. Use shorthand @ID (e.g., @0, @17363629609536820). Full anchor format also works.
2. End new entry content with hashtags on their own line.
3. Use markdown formatting, including $ and $$ for math expressions.
4. Reference other entries with [label](#ID) format.

Style considerations:
1. Write in first-person POV so entries stand alone without conversation context.
2. Prefer prose over lists (max 3 bullet points).
3. Connect new entries to parent content; short quotes can make links clear.
4. To add an image use this format `<img src="{absolute_path}" alt="{file_name}" style="zoom:67%;" />`
"""

# ---------------------------------------------------------------------------
# Composed specifications -- these are what consumers import
# ---------------------------------------------------------------------------

Scratch_Instructions = _Shared_Format + _Scratch_Addendum
Code_Instructions = _Shared_Format + _Code_Addendum
Proposal_Instructions = _Shared_Format + _Entry_Addendum


# ===================================================================
# config.py — API key stub (populate from environment or config file)
# ===================================================================

def _load_config():
    """Load API keys from environment variables or config.py if available."""
    config = {
        "ANTHROPIC_API_KEYS": {},
        "DEFAULT_API_KEY_TYPE": "research",
        "ANTHROPIC_MODEL": {},
        "OPENAI_API_KEY": None,
        "OPENAI_MODEL": "o3-mini",
        "MAX_TOKENS": 16384,
    }
    
    # Try environment variables first
    research_key = os.environ.get("ANTHROPIC_API_KEY_RESEARCH") or os.environ.get("ANTHROPIC_API_KEY")
    personal_key = os.environ.get("ANTHROPIC_API_KEY_PERSONAL")
    
    if research_key:
        config["ANTHROPIC_API_KEYS"]["research"] = research_key
    if personal_key:
        config["ANTHROPIC_API_KEYS"]["personal"] = personal_key
    
    config["ANTHROPIC_MODEL"] = {
        "research": os.environ.get("ANTHROPIC_MODEL_RESEARCH", "claude-sonnet-4-20250514"),
        "personal": os.environ.get("ANTHROPIC_MODEL_PERSONAL", "claude-sonnet-4-20250514"),
    }
    
    config["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
    config["MAX_TOKENS"] = int(os.environ.get("MAX_TOKENS", "16384"))
    
    # Try importing config.py as fallback
    try:
        import config as cfg
        if hasattr(cfg, "ANTHROPIC_API_KEYS"):
            config["ANTHROPIC_API_KEYS"] = cfg.ANTHROPIC_API_KEYS
        if hasattr(cfg, "DEFAULT_API_KEY_TYPE"):
            config["DEFAULT_API_KEY_TYPE"] = cfg.DEFAULT_API_KEY_TYPE
        if hasattr(cfg, "ANTHROPIC_MODEL"):
            config["ANTHROPIC_MODEL"] = cfg.ANTHROPIC_MODEL
        if hasattr(cfg, "OPENAI_API_KEY"):
            config["OPENAI_API_KEY"] = cfg.OPENAI_API_KEY
        if hasattr(cfg, "OPENAI_MODEL"):
            config["OPENAI_MODEL"] = cfg.OPENAI_MODEL
        if hasattr(cfg, "MAX_TOKENS"):
            config["MAX_TOKENS"] = cfg.MAX_TOKENS
    except ImportError:
        pass
    
    return config

_CONFIG = _load_config()
ANTHROPIC_API_KEYS = _CONFIG["ANTHROPIC_API_KEYS"]
DEFAULT_API_KEY_TYPE = _CONFIG["DEFAULT_API_KEY_TYPE"]
ANTHROPIC_MODEL = _CONFIG["ANTHROPIC_MODEL"]
OPENAI_API_KEY = _CONFIG["OPENAI_API_KEY"]
OPENAI_MODEL = _CONFIG["OPENAI_MODEL"]
MAX_TOKENS = _CONFIG["MAX_TOKENS"]


# ===================================================================
# document_utils.py — generate_timestamp only
# ===================================================================

def generate_timestamp():
    """Generate a timestamp for entry IDs (10^7 scale to match create_new_entry)"""
    return str(int(time.time() * 10_000_000))


# ===================================================================
# LLM_methods.py — ConversationManager, generate_completion, venv detection
# ===================================================================

class ConversationManager:
    def __init__(self):
        self.root = {"children": []}
        self.current_path = []
        self.start_entry_id = None
        self.code_files = []
        self.conversation_id = None
        self.uploaded_files = []
        self.conversation_name = "Untitled Conversation"
        self.last_updated = None
        self.api_key_type = "research"  # Default to research
        self.selected_project = "playground/temp"  # Default project
        # Context settings
        self.chat_mode = "base"
        self.context_scope = "local"  # "local" or "global"
        self.context_token_budget = 10000
        self.max_history = 50
        # Agent iteration settings
        self.max_parts = 3  # Auto-continuation limit
        self.max_output_chars = 20000  # Execution output truncation
        self.exec_timeout = 90  # Seconds before execution timeout

    def start_new_conversation(
        self, start_entry_id, code_files, name="Untitled Conversation"
    ):
        self.root = {"children": []}
        self.current_path = []
        self.start_entry_id = start_entry_id
        self.code_files = code_files
        self.conversation_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.uploaded_files = []
        self.conversation_name = name
        self.last_updated = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    def add_uploaded_file(self, name, content):
        # Check if a file with the same name already exists
        if not any(file["name"] == name for file in self.uploaded_files):
            self.uploaded_files.append({"name": name, "content": content})

    def remove_uploaded_file(self, file_name):
        self.uploaded_files = [
            file for file in self.uploaded_files if file["name"] != file_name
        ]

    def get_uploaded_files(self):
        return self.uploaded_files

    def get_conversation_history(self):
        history = []
        node = self.root
        for index in self.current_path:
            node = node["children"][index]
            if node is not []:
                history.append({"user": node["user"], "assistant": node["assistant"]})
        return history

    def add_turn(self, user_input, assistant_response, modifications, action=None, turn_state=None):
        new_node = {
            "user": user_input,
            "assistant": assistant_response,
            "action": action,
            "modifications": modifications,
            "applied_modifications": [],
            "turn_state": turn_state,  # Structured state snapshot at time of turn
            "children": [],
        }
        current_node = self.get_current_node()
        current_node["children"].append(new_node)
        self.current_path.append(len(current_node["children"]) - 1)
        
        # Update timestamp when adding new turn
        self.last_updated = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    def get_current_node(self):
        node = self.root
        for index in self.current_path:
            node = node["children"][index]
        return node

    def get_current_turn(self):
        return self.get_current_node()

    def update_current_turn(self, new_turn):
        node = self.root
        for index in self.current_path:
            node = node["children"][index]
        node = new_turn

    def move(self, direction):
        if direction == "back":
            if self.current_path:
                self.current_path.pop()
        elif direction == "forward":
            current_node = self.get_current_node()
            if current_node["children"]:
                self.current_path.append(0)
        elif direction == "up" and self.current_path:
            parent = self.root
            for index in self.current_path[:-1]:
                parent = parent["children"][index]
            if self.current_path[-1] > 0:
                self.current_path[-1] -= 1
        elif direction == "down" and self.current_path:
            parent = self.root
            for index in self.current_path[:-1]:
                parent = parent["children"][index]
            if self.current_path[-1] < len(parent["children"]) - 1:
                self.current_path[-1] += 1

    def get_sibling_count(self):
        if not self.current_path:
            return len(self.root["children"])
        parent = self.root
        for index in self.current_path[:-1]:
            parent = parent["children"][index]
        return len(parent["children"])

    def get_current_branch_index(self):
        return self.current_path[-1] if self.current_path else -1

    def has_messages(self):
        return len(self.root["children"]) > 0

    def clean_for_serialization(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                if key == "graph":
                    data[key] = (
                        None  # or you could store some basic info about the graph if needed
                    )
                elif isinstance(value, (dict, list)):
                    data[key] = self.clean_for_serialization(value)
        elif isinstance(data, list):
            data = [self.clean_for_serialization(item) for item in data]
        return data

    def to_dict(self):
        # Update timestamp before saving
        if not self.last_updated:
            self.last_updated = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            
        clean_data = self.clean_for_serialization(
            copy.deepcopy(
                {
                    "root": self.root,
                    "current_path": self.current_path,
                    "start_entry_id": self.start_entry_id,
                    "code_files": self.code_files,
                    "conversation_id": self.conversation_id,
                    "uploaded_files": self.uploaded_files,
                    "conversation_name": self.conversation_name,
                    "last_updated": self.last_updated,
                    "api_key_type": self.api_key_type,
                    "selected_project": self.selected_project,
                    # Context settings
                    "chat_mode": self.chat_mode,
                    "context_scope": self.context_scope,
                    "context_token_budget": self.context_token_budget,
                    "max_history": self.max_history,
                    # Agent iteration settings
                    "max_parts": self.max_parts,
                    "max_output_chars": self.max_output_chars,
                    "exec_timeout": self.exec_timeout,
                }
            )
        )
        return clean_data

    @classmethod
    def from_dict(cls, data):
        manager = cls()
        manager.root = data["root"]
        manager.current_path = data["current_path"]
        manager.start_entry_id = data["start_entry_id"]
        manager.code_files = data["code_files"]
        manager.conversation_id = data["conversation_id"]
        manager.uploaded_files = data.get("uploaded_files", [])
        manager.conversation_name = data.get(
            "conversation_name", "Untitled Conversation"
        )
        manager.last_updated = data.get("last_updated")
        manager.api_key_type = data.get("api_key_type", "research")
        manager.selected_project = data.get("selected_project", "scripts")
        # Context settings
        manager.chat_mode = data.get("chat_mode", "base")
        manager.context_scope = data.get("context_scope", "local")
        manager.context_token_budget = data.get("context_token_budget", 10000)
        manager.max_history = data.get("max_history", 50)
        # Agent iteration settings
        manager.max_parts = data.get("max_parts", 3)
        manager.max_output_chars = data.get("max_output_chars", 20000)
        manager.exec_timeout = data.get("exec_timeout", 90)
        return manager

    def reset_to_root(self):
        self.current_path = []


# Graph Operations

def format_messages(prompt, conversation_history):
    messages = []
    for turn in conversation_history:
        # Only add messages with non-empty content
        user_content = turn.get("user") if turn.get("user") else ""
        assistant_content = turn.get("assistant") if turn.get("assistant") else ""
        
        # Convert to string and strip whitespace
        user_content = str(user_content).strip() if user_content else ""
        assistant_content = str(assistant_content).strip() if assistant_content else ""
        
        if user_content:
            messages.append({"role": "user", "content": user_content})
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})
    
    # Always add the current prompt if non-empty
    prompt_content = str(prompt).strip() if prompt else ""
    if prompt_content:
        messages.append({"role": "user", "content": prompt_content})
    
    # Final safety check - remove any messages with empty content
    messages = [m for m in messages if m.get("content") and str(m["content"]).strip()]
    
    return messages

def generate_completion(prompt, system_prompt, conversation_history, provider="anthropic", api_key_type="research", model_name=None, usage_callback=None):
    """
    Streams completions from Anthropic or OpenAI.
    - provider: set to "Anthropic" (default) or "OpenAI".
    - api_key_type: "research" or "personal" (only used for Anthropic)
    - model_name: explicit model override (e.g. "claude-opus-4-5-20251101"). If None, uses config.
    - usage_callback: callable(dict) called after streaming with usage stats:
        {"input_tokens": int, "output_tokens": int, "cache_creation_input_tokens": int, "cache_read_input_tokens": int}
    """
    messages = format_messages(prompt, conversation_history)
    total_text = system_prompt + "\n" + "\n".join(m["content"] for m in messages)
    
    # Check overall token count.
    if provider.lower() == "anthropic":
        try:
            # Get the appropriate API key
            api_key = ANTHROPIC_API_KEYS.get(api_key_type) or ANTHROPIC_API_KEYS.get(DEFAULT_API_KEY_TYPE)
            if not api_key:
                available = list(ANTHROPIC_API_KEYS.keys()) or ["none"]
                yield f"Error: No Anthropic API key found for '{api_key_type}'. Available key types: {available}. Set ANTHROPIC_API_KEY environment variable."
                return
            client = anthropic.Anthropic(api_key=api_key)
            model_type = model_name or ANTHROPIC_MODEL.get(api_key_type, "claude-sonnet-4-20250514")

            with client.messages.stream(
                model=model_type,
                system=system_prompt,
                messages=messages,
                max_tokens=MAX_TOKENS,
            ) as stream:
                for text in stream.text_stream:
                    yield text
                # Capture usage after stream completes
                if usage_callback:
                    try:
                        final = stream.get_final_message()
                        usage = final.usage
                        usage_callback({
                            "input_tokens": getattr(usage, "input_tokens", 0),
                            "output_tokens": getattr(usage, "output_tokens", 0),
                            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
                            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
                        })
                    except Exception as ue:
                        print(f"Warning: could not capture usage: {ue}")
        except Exception as e:
            print(f"Error during streaming completion (Anthropic): {e}")
            yield f"Error: {str(e)}"
    
    elif provider.lower() == "openai":
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            token_count = count_tokens_openai(total_text, OPENAI_MODEL)
            if token_count >= 200000:
                yield "Error: Context length exceeds 200,000 tokens. Please reduce the input size."
                return

            # Build messages with the system prompt as the first message.
            messages_for_api = [{"role": "system", "content": system_prompt}] + messages
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                reasoning_effort = "high",
                messages=messages_for_api,
                max_completion_tokens=MAX_TOKENS,
                stream=True,
            )
            for chunk in response:
                # Each chunk has choices; yield content if available.
                delta = chunk.choices[0].delta
                content = delta.content if delta and delta.content is not None else ""
                yield content
        except Exception as e:
            print(f"Error during streaming completion (OpenAI): {e}")
            yield f"Error: {str(e)}"
    
    else:
        yield "Error: Unknown provider specified."

def detect_project_venv(project_path: str) -> dict:
    """
    Detect virtual environment for a project.
    Returns dict with python path and activation info.
    """
    # Common venv directory names
    venv_names = ['venv', '.venv', 'env', '.env']
    
    for venv_name in venv_names:
        venv_path = os.path.join(project_path, venv_name)
        if os.path.isdir(venv_path):
            if os.name == 'nt':  # Windows
                python_path = os.path.join(venv_path, 'Scripts', 'python.exe')
                activate_path = os.path.join(venv_path, 'Scripts', 'activate.bat')
            else:  # Unix
                python_path = os.path.join(venv_path, 'bin', 'python')
                activate_path = os.path.join(venv_path, 'bin', 'activate')
            
            if os.path.exists(python_path):
                return {
                    'python': python_path,
                    'activate': activate_path,
                    'venv_path': venv_path,
                    'detected': True
                }
    
    # No venv found - use system python
    return {
        'python': 'python',
        'activate': None,
        'venv_path': None,
        'detected': False
    }

def get_project_env(project_path: str) -> dict:
    """
    Get environment variables for running commands in a project.
    """
    venv_info = detect_project_venv(project_path)
    env = os.environ.copy()
    
    if venv_info['detected']:
        # Modify PATH to prioritize venv
        venv_bin = os.path.dirname(venv_info['python'])
        env['PATH'] = venv_bin + os.pathsep + env.get('PATH', '')
        env['VIRTUAL_ENV'] = venv_info['venv_path']
        # Remove PYTHONHOME if set (can interfere with venv)
        env.pop('PYTHONHOME', None)
    
    return env

# ===================================================================
# code_edit_utils.py — SEARCH/REPLACE parsing, execution, error detection
# ===================================================================

FILE_LINE_RE = re.compile(r"^\s*([^\n]+?\.(?:py|md|tex|bib|yml|yaml|json|txt))\s*$")

SEARCH_MARKER = "<<<<<<< SEARCH"
SEP_MARKER = "======="
REPLACE_MARKER = ">>>>>>> REPLACE"
CODE_FENCE = "```"


def parse_search_replace_blocks(
    response: str, context_files: List[str] = None, track_incomplete: bool = False
) -> List[Dict]:
    """
    Parse SEARCH/REPLACE blocks from LLM response, correctly handling nested delimiter markers
    inside SEARCH/REPLACE content by tracking nesting depth.

    Key behaviors:
    - Outside blocks: skips markdown code-fence lines like ``` or ```python or ```diff (formatting noise).
    - Inside blocks: preserves code fences as literal content (so replacements can include fenced examples).
    - Nested delimiter markers inside payload are treated as literal content via depth tracking.
    - If track_incomplete=True, incomplete blocks are included with "incomplete": True flag.

    Expected block format:
        file/path/here.py
        <<<<<<< SEARCH
        exact lines to find
        =======
        replacement lines
        >>>>>>> REPLACE
    """
    lines = response.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    basename_to_path = {}
    if context_files:
        for full_path in context_files:
            basename_to_path[os.path.basename(full_path)] = full_path

    def _clean_file_line(s: str) -> str:
        # Remove accidental code fence artifacts on same line (rare but harmless)
        return s.replace(CODE_FENCE, "").strip()

    def _is_fence_outside(s: str) -> bool:
        # Treat ``` and ```lang as fences when scanning for blocks
        return s.lstrip().startswith("```")

    blocks: List[Dict] = []
    i = 0
    n = len(lines)

    while i < n:
        # Outside blocks: skip fence lines entirely
        if _is_fence_outside(lines[i]):
            i += 1
            continue

        m = FILE_LINE_RE.match(lines[i])
        if not m:
            i += 1
            continue

        file_path_raw = _clean_file_line(m.group(1))

        # Next non-fence line must be SEARCH marker
        j = i + 1
        while j < n and _is_fence_outside(lines[j]):
            j += 1
        if j >= n or lines[j].strip() != SEARCH_MARKER:
            i += 1
            continue

        file_path = _resolve_file_path(file_path_raw, basename_to_path)

        search_buf: List[str] = []
        replace_buf: List[str] = []
        state = "SEARCH"
        depth = 0
        found_closing = False

        k = j + 1
        while k < n:
            cur = lines[k]
            cur_stripped = cur.strip()

            # IMPORTANT: inside blocks, do NOT skip fences; they are valid content.

            if state == "SEARCH":
                if cur_stripped == SEARCH_MARKER:
                    depth += 1
                    search_buf.append(cur)
                elif cur_stripped == REPLACE_MARKER:
                    # Closes nested blocks inside SEARCH if depth>0; otherwise literal
                    if depth > 0:
                        depth -= 1
                    search_buf.append(cur)
                elif cur_stripped == SEP_MARKER and depth == 0:
                    state = "REPLACE"
                else:
                    search_buf.append(cur)

            else:  # state == "REPLACE"
                if cur_stripped == SEARCH_MARKER:
                    depth += 1
                    replace_buf.append(cur)
                elif cur_stripped == REPLACE_MARKER:
                    if depth == 0:
                        found_closing = True
                        k += 1  # consume closing marker
                        break
                    depth -= 1
                    replace_buf.append(cur)
                else:
                    replace_buf.append(cur)

            k += 1

        # If we never found the closing marker, don't produce a half-baked block
        if not found_closing:
            if track_incomplete:
                blocks.append(
                    {
                        "file_path": file_path,
                        "search": "\n".join(search_buf),
                        "replace": "\n".join(replace_buf),
                        "incomplete": True,
                    }
                )
            i = j + 1
            continue

        blocks.append(
            {
                "file_path": file_path,
                "search": "\n".join(search_buf),
                "replace": "\n".join(replace_buf),
                "incomplete": False,
            }
        )

        i = k

    return blocks


def _resolve_file_path(file_path: str, basename_to_path: dict) -> str:
    if os.path.exists(file_path):
        return file_path
    basename = os.path.basename(file_path)
    if basename in basename_to_path:
        return basename_to_path[basename]
    return file_path


def run_scratch_script(project_dir: str, timeout: int = 90, max_chars: int = 20000) -> str:
    """
    Execute scratch.py in the given project directory.
    Returns formatted output string (stdout + stderr), truncated if needed.
    """
    import subprocess
        
    # Ensure project_dir is absolute for reliable path resolution
    project_dir = os.path.abspath(project_dir)
    
    script_path = os.path.join(project_dir, "scratch.py")
    if not os.path.exists(script_path):
        return f"Error: {script_path} not found"
    
    env = get_project_env(project_dir)
    
    # Determine python executable
    if env.get("VIRTUAL_ENV"):
        if os.name == 'nt':  # Windows
            python_exe = os.path.join(env["VIRTUAL_ENV"], "Scripts", "python.exe")
        else:
            python_exe = os.path.join(env["VIRTUAL_ENV"], "bin", "python")
    else:
        python_exe = "python"
    
    try:
        result = subprocess.run(
            [python_exe, "scratch.py"],
            cwd=project_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += result.stderr
        if result.returncode != 0:
            output = f"[Exit code: {result.returncode}]\n{output}"
    except subprocess.TimeoutExpired:
        output = f"[Timeout: exceeded {timeout}s]"
    except Exception as e:
        output = f"[Execution error: {str(e)}]"
    
    if len(output) > max_chars:
        output = output[:max_chars] + f"\n... [truncated, {len(output)} chars total]"
    
    return output


def has_runtime_error(output: str) -> bool:
    """
    Check if execution output indicates a runtime error.
    Returns True if there's a traceback, non-zero exit code, or execution error.
    """
    if not output:
        return False
    
    error_indicators = [
        "Traceback (most recent call last):",
        "[Exit code:",
        "[Execution error:",
        "[Timeout:",
        "Error:",
        "Exception:",
        "SyntaxError:",
        "NameError:",
        "TypeError:",
        "ValueError:",
        "AttributeError:",
        "ImportError:",
        "ModuleNotFoundError:",
        "KeyError:",
        "IndexError:",
        "ZeroDivisionError:",
        "FileNotFoundError:",
        "RuntimeError:",
    ]
    
    for indicator in error_indicators:
        if indicator in output:
            return True
    
    return False


def extract_error_summary(output: str, max_length: int = 200) -> str:
    """
    Extract a brief error summary from execution output.
    Returns the most relevant error line, truncated if needed.
    """
    if not output:
        return "unknown error"
    
    lines = output.strip().split("\n")
    
    # Look for specific error patterns
    for i, line in enumerate(lines):
        # Check for common Python exception lines
        if any(err in line for err in ["Error:", "Exception:", "Timeout"]):
            # Include the error line and maybe one line of context
            error_line = line.strip()
            if len(error_line) > max_length:
                error_line = error_line[:max_length] + "..."
            return error_line
    
    # Check for exit code
    for line in lines:
        if "[Exit code:" in line:
            return line.strip()
    
    # Fallback: return last non-empty line
    for line in reversed(lines):
        if line.strip():
            result = line.strip()
            if len(result) > max_length:
                result = result[:max_length] + "..."
            return result
    
    return "execution failed"





# ===================================================================
# edit_models.py — Edit class hierarchy, intent, continuation logic
# ===================================================================

# ---------------------------------------------------------------------------
# Intent and result enums
# ---------------------------------------------------------------------------

class EditIntent(Enum):
    """What the LLM expects to happen after this edit is applied."""
    APPLY_ONLY = auto()       # Apply edit, no execution, no continuation
    EXECUTE = auto()          # Apply + execute scratch.py, show output, continue
    EXECUTE_FINAL = auto()    # Apply + execute, show output, do NOT continue
    PROVISIONAL = auto()      # Stage for human review (entry mode)


class EditResult(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    INCOMPLETE = auto()       # Missing >>>>>>> REPLACE marker


# ---------------------------------------------------------------------------
# EditReport — uniform result from any apply() call
# ---------------------------------------------------------------------------

@dataclass
class EditReport:
    """Uniform result from any apply() call."""
    result: EditResult
    message: str
    file_path: Optional[str] = None
    entry_id: Optional[str] = None
    created: bool = False

    @property
    def success(self) -> bool:
        return self.result == EditResult.SUCCESS


# ---------------------------------------------------------------------------
# Base Edit class
# ---------------------------------------------------------------------------

@dataclass
class Edit:
    """
    Base class for all parsed SEARCH/REPLACE blocks.

    The parser populates these; the loop consumes them via
    resolve() → apply(). Subclasses implement _do_resolve()
    and _do_apply() for type-specific logic.
    """
    search: str = ""
    replace: str = ""
    incomplete: bool = False
    intent: EditIntent = EditIntent.APPLY_ONLY

    # Set by resolve()
    _resolved: bool = field(default=False, repr=False)
    _resolve_error: Optional[str] = field(default=None, repr=False)
    requires_user_action: bool = field(default=False, repr=False)

    def resolve(self, **context) -> bool:
        """
        Validate and resolve targets. Returns True on success.
        Subclasses override _do_resolve() rather than this method.
        """
        if self.incomplete:
            self._resolved = True
            self._resolve_error = "Incomplete block - missing >>>>>>> REPLACE marker"
            return False
        self._resolved = True
        return self._do_resolve(**context)

    def _do_resolve(self, **context) -> bool:
        """Override in subclasses for type-specific resolution."""
        return True

    def apply(self) -> EditReport:
        """Apply the edit. Must call resolve() first."""
        if not self._resolved:
            return EditReport(EditResult.FAILURE, "resolve() not called")
        if self.incomplete:
            return EditReport(
                EditResult.INCOMPLETE,
                f"Incomplete SEARCH/REPLACE block - missing >>>>>>> REPLACE marker. If running out of space, break large edits into smaller pieces.",
            )
        if self._resolve_error:
            return EditReport(EditResult.FAILURE, self._resolve_error)
        return self._do_apply()

    def _do_apply(self) -> EditReport:
        """Override in subclasses for type-specific application."""
        raise NotImplementedError

    @property
    def expects_execution(self) -> bool:
        return self.intent in (EditIntent.EXECUTE, EditIntent.EXECUTE_FINAL)

    @property
    def expects_continuation(self) -> bool:
        return self.intent == EditIntent.EXECUTE


# ---------------------------------------------------------------------------
# CodeEdit — targets a file on disk
# ---------------------------------------------------------------------------

@dataclass
class CodeEdit(Edit):
    """Edit targeting a code/text file."""
    raw_path: str = ""
    resolved_path: str = ""

    def _do_resolve(self, *, context_files=None, project_dir=None,
                    scratch_only=False, allowed_abs=None) -> bool:
        """
        Resolve raw_path to an absolute file path.

        Args:
            context_files: List of full paths the LLM can see/edit.
            project_dir: Base directory for new file creation.
            scratch_only: If True, reject anything that isn't scratch.py.
            allowed_abs: Optional set of absolute paths that are editable.
                         If provided, resolved path must be in this set
                         (or be a new file in project_dir, or scratch.py).
        """
        basename = os.path.basename(self.raw_path)

        # scratch_only filtering
        if scratch_only and basename != "scratch.py":
            self._resolve_error = f"Base mode: only scratch.py allowed, got {self.raw_path}. Stop and ask the user how to proceed."
            self.requires_user_action = True
            return False

        # Build basename→full_path map
        basename_to_path = {}
        if context_files:
            for fp in context_files:
                basename_to_path[os.path.basename(fp)] = fp

        search_stripped = (self.search or "").strip()
        is_new_or_replace = not search_stripped or search_stripped == basename

        # Resolution order: exact path → basename match → new file in project_dir
        if scratch_only and project_dir:
            self.resolved_path = os.path.join(project_dir, "scratch.py")
        elif os.path.exists(self.raw_path):
            self.resolved_path = self.raw_path
        elif basename in basename_to_path:
            self.resolved_path = basename_to_path[basename]
        elif is_new_or_replace and project_dir:
            self.resolved_path = os.path.join(project_dir, basename)
        else:
            self._resolve_error = f"Cannot resolve path: {self.raw_path}"
            return False

        # Allowlist check (edit mode)
        if allowed_abs is not None and not scratch_only:
            abs_resolved = os.path.abspath(self.resolved_path)
            is_scratch = (basename == "scratch.py")
            is_in_project = project_dir and os.path.abspath(project_dir) in abs_resolved
            if abs_resolved not in allowed_abs and not is_scratch and not (is_new_or_replace and is_in_project):
                self._resolve_error = f"File not in editable set: {self.raw_path}. Stop and ask the user if they want this file to be edited."
                self.requires_user_action = True
                return False

        return True

    def _do_apply(self) -> EditReport:
        """Apply edit to the resolved file path."""
        success, message = _apply_file_edit(self.resolved_path, self.search, self.replace)
        return EditReport(
            result=EditResult.SUCCESS if success else EditResult.FAILURE,
            message=message,
            file_path=self.resolved_path,
            created=success and "Created new file" in message,
        )

    @property
    def is_scratch(self) -> bool:
        return os.path.basename(self.resolved_path or self.raw_path) == "scratch.py"


# ---------------------------------------------------------------------------
# EntryEdit — targets a document tree entry
# ---------------------------------------------------------------------------

@dataclass
class EntryEdit(Edit):
    """Edit targeting a document tree entry."""
    operation: str = "create"   # "create" or "update"
    entry_id: str = ""
    parent_id: Optional[str] = None

    # Set by resolve()
    entry_exists: Optional[bool] = field(default=None, repr=False)
    parent_exists: Optional[bool] = field(default=None, repr=False)

    def __post_init__(self):
        # Entry edits are always provisional — human must confirm
        if self.intent == EditIntent.APPLY_ONLY:
            self.intent = EditIntent.PROVISIONAL

    def _do_resolve(self, *, graph=None, **kwargs) -> bool:
        """Check that the target entry/parent exists in the graph."""
        if graph is None:
            self._resolve_error = "No graph provided"
            return False

        if self.operation == "update":
            self.entry_exists = self.entry_id in graph.nodes
            if not self.entry_exists:
                self._resolve_error = f"Entry '{self.entry_id}' does not exist"
                return False
        elif self.operation == "create":
            self.parent_exists = (
                self.parent_id is not None and self.parent_id in graph.nodes
            )
            if not self.parent_exists:
                self._resolve_error = f"Parent '{self.parent_id}' does not exist"
                return False

        return True

    def _do_apply(self) -> EditReport:
        """
        Stage the entry modification. Does NOT commit to graph —
        entry mode requires human confirmation via sidebar.apply_entry_modification().
        """
        op_desc = f"Staged {self.operation} for entry '{self.entry_id}'"
        if self.parent_id:
            op_desc += f" (parent: '{self.parent_id}')"
        return EditReport(
            result=EditResult.SUCCESS,
            message=op_desc,
            entry_id=self.entry_id,
            created=(self.operation == "create"),
        )

    def to_modification_dict(self) -> dict:
        """
        Convert to the dict format expected by sidebar.apply_entry_modification().
        Bridges the new Edit model to the existing entry application interface.
        """
        d = {
            "entry_id": self.entry_id,
            "parent_id": self.parent_id,
            "entry_content": self.replace,
            "type": "entry",
            "operation": self.operation,
            "search": self.search,
            "parent_exists": self.parent_exists,
            "entry_exists": self.entry_exists,
        }
        return d


# ---------------------------------------------------------------------------
# File I/O — extracted from code_edit_utils.apply_edit()
# ---------------------------------------------------------------------------

def _apply_file_edit(file_path: str, search: str, replace: str) -> Tuple[bool, str]:
    """
    Apply a single file edit. Identical logic to code_edit_utils.apply_edit().
    Kept here so CodeEdit._do_apply() has no cross-module dependency.
    """
    search_stripped = search.strip()
    if not search_stripped or search_stripped == os.path.basename(file_path):
        try:
            dir_path = os.path.dirname(file_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            with open(file_path, "w") as f:
                f.write(replace)
            return True, f"Created new file: {file_path}"
        except Exception as e:
            return False, f"Failed to create {file_path}: {str(e)}"

    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            content = f.read()
    except Exception as e:
        return False, f"Failed to read {file_path}: {str(e)}"

    if search not in content:
        search_lines = search.strip().split("\n")
        first_line = search_lines[0].strip() if search_lines else ""
        if first_line and first_line in content:
            return False, f"SEARCH failed in {file_path}: First line found but full block doesn't match (check whitespace/indentation)"
        else:
            return False, f"SEARCH failed in {file_path}: Content not found"

    count = content.count(search)
    if count > 1:
        return False, f"SEARCH failed in {file_path}: Found {count} matches (need exactly 1)"

    new_content = content.replace(search, replace, 1)

    try:
        with open(file_path, "w") as f:
            f.write(new_content)
        search_lines = len(search.split("\n"))
        replace_lines = len(replace.split("\n"))
        return True, f"Applied edit to {file_path}: {search_lines} lines → {replace_lines} lines"
    except Exception as e:
        return False, f"Failed to write {file_path}: {str(e)}"


# ---------------------------------------------------------------------------
# Intent inference — the seam for future explicit intent parsing
# ---------------------------------------------------------------------------

def infer_code_intent(file_path: str) -> EditIntent:
    """
    Default intent inference for code edits.
    scratch.py → EXECUTE (expects continuation).
    Everything else → APPLY_ONLY.

    This is the baseline heuristic. Future: parse explicit intent markers
    from the LLM response (e.g. "# INTENT: execute_final").
    """
    if os.path.basename(file_path) == "scratch.py":
        return EditIntent.EXECUTE
    return EditIntent.APPLY_ONLY


# ---------------------------------------------------------------------------
# Continuation logic — replaces get_continuation_reason()
# ---------------------------------------------------------------------------

def should_continue(edits: list, reports: list, exec_output: str) -> Tuple[Optional[str], str]:
    """
    Intent-driven continuation decision. Replaces get_continuation_reason().

    Reads intent from Edit objects rather than inferring from output signals.

    Returns:
        (reason_type, detail) where reason_type is one of:
        - None: no continuation needed
        - "continuing": execution completed, agent expects to see output
        - "edit_error": an edit failed to apply
        - "runtime_error": execution produced an error
        - "timeout": execution timed out
    """
    if not edits or not reports:
        return (None, "")

    # Check if any failure requires user intervention (don't auto-continue)
    user_action_needed = any(
        getattr(e, 'requires_user_action', False) for e in edits
    )
    if user_action_needed:
        return (None, "")

    # Any other edit failure triggers continuation
    failures = [r for r in reports if not r.success]
    if failures:
        return ("edit_error", failures[0].message)

    # Check execution intent
    executed_edits = [e for e in edits if e.expects_execution and e._resolved and not e._resolve_error]
    if not executed_edits:
        return (None, "")

    if not exec_output:
        return (None, "")

    # Check if any edit wants to continue after execution
    wants_continuation = any(e.expects_continuation for e in executed_edits)
    if not wants_continuation:
        # EXECUTE_FINAL: ran, but don't continue
        return (None, "")

    # EXECUTE with output: check for errors
    if "[Timeout:" in exec_output:
        return ("timeout", "execution exceeded time limit")

    if has_runtime_error(exec_output):
        return ("runtime_error", extract_error_summary(exec_output))

    return ("continuing", "execution completed")


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------

def resolve_all(edits: list, **context) -> list:
    """Resolve all edits, returning the same list (mutated in place)."""
    for edit in edits:
        if isinstance(edit, CodeEdit):
            edit.resolve(
                context_files=context.get("context_files"),
                project_dir=context.get("project_dir"),
                scratch_only=context.get("scratch_only", False),
                allowed_abs=context.get("allowed_abs"),
            )
        elif isinstance(edit, EntryEdit):
            edit.resolve(graph=context.get("graph"))
        else:
            edit.resolve(**context)
    return edits


def apply_all(edits: list) -> list:
    """Apply all edits, returning list of EditReport."""
    return [edit.apply() for edit in edits]


# ---------------------------------------------------------------------------
# Factory: parse LLM response → typed Edit objects
# ---------------------------------------------------------------------------

def parse_edits(
    response: str,
    *,
    mode: str = "base",
    context_files: List[str] = None,
    graph=None,
) -> List[Edit]:
    """
    Parse an LLM response into typed Edit objects.

    This is the single entry point for all SEARCH/REPLACE parsing.
    Inspects block targets to produce the right Edit subclass:
    - File paths → CodeEdit
    - Entry targets (@{id}, "new") → EntryEdit

    Args:
        response: Raw LLM response text.
        mode: One of "base", "edit", "entry". Controls which parsers run.
        context_files: List of editable file paths (for code edit resolution).
        graph: NetworkX DiGraph (for entry edit resolution).
    """
    edits: List[Edit] = []

    if mode in ("base", "edit"):
        blocks = parse_search_replace_blocks(
            response,
            context_files=context_files if mode == "edit" else [],
            track_incomplete=True,
        )
        for block in blocks:
            edit = CodeEdit(
                raw_path=block.get("file_path", ""),
                search=block.get("search", ""),
                replace=block.get("replace", ""),
                incomplete=block.get("incomplete", False),
                intent=infer_code_intent(block.get("file_path", "")),
            )
            edits.append(edit)

    elif mode == "entry":
        sr_blocks = parse_entry_search_replace_blocks(response, graph=graph)
        for block in sr_blocks:
            if block["operation"] == "create":
                edit = EntryEdit(
                    operation="create",
                    entry_id=block["entry_id"],
                    parent_id=block.get("parent_id"),
                    search="",
                    replace=block["replace"],
                    intent=EditIntent.PROVISIONAL,
                    parent_exists=block.get("parent_exists"),
                )
                edits.append(edit)
            else:
                edit = EntryEdit(
                    operation="update",
                    entry_id=block["entry_id"],
                    search=block.get("search", ""),
                    replace=block["replace"],
                    intent=EditIntent.PROVISIONAL,
                    entry_exists=block.get("entry_exists"),
                )
                edits.append(edit)

    return edits


def format_edit_log(reports: list) -> str:
    """Format EditReport list for display. Same output as code_edit_utils.format_edit_log."""
    if not reports:
        return "No SEARCH/REPLACE blocks found in response."

    lines = []
    success_count = sum(1 for r in reports if r.success)
    fail_count = len(reports) - success_count

    lines.append(f"## Edit Results: {success_count} applied, {fail_count} failed\n")

    for r in reports:
        status = "✓" if r.success else "✗"
        lines.append(f"{status} {r.message}")

    return "\n".join(lines)

# ===================================================================
# conversation_storage.py — save_conversation, load_conversation
# ===================================================================

CONVERSATIONS_DIR = 'conversations'

def ensure_conversations_dir():
    if not os.path.exists(CONVERSATIONS_DIR):
        os.makedirs(CONVERSATIONS_DIR)

def save_conversation(conversation_manager):
    ensure_conversations_dir()
    
    if not conversation_manager.has_messages():
        filename = f"{conversation_manager.conversation_id}.json"
        filepath = os.path.join(CONVERSATIONS_DIR, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        return

    conversation_data = conversation_manager.to_dict()  # This now returns cleaned data
    filename = f"{conversation_manager.conversation_id}.json"
    filepath = os.path.join(CONVERSATIONS_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(conversation_data, f)

def load_conversation(conversation_id):
    filepath = os.path.join(CONVERSATIONS_DIR, f"{conversation_id}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                conversation_data = json.load(f)
            loaded_conversation = ConversationManager.from_dict(conversation_data)
            
            # Check if the loaded conversation has any messages
            if not loaded_conversation.has_messages():
                os.remove(filepath)  # Remove the empty conversation file
                return None
            
            return loaded_conversation
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {filepath}")
        except AttributeError:
            print(f"Error: ConversationManager.from_dict() method not found")
        except Exception as e:
            print(f"Error loading conversation: {str(e)}")
    return None