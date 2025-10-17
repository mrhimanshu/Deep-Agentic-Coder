import os
import re
import json
import asyncio
import difflib
import subprocess
from importlib.metadata import version
from dotenv import load_dotenv
from typing import Annotated, List, Literal, Union, Dict, Sequence, Optional, Any, ClassVar
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages

from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from langgraph.prebuilt.chat_agent_executor import AgentState as LGAgentState, StateGraph

from pydantic import BaseModel, Field

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.text import Text
from rich.syntax import Syntax

import prompt

load_dotenv(os.path.join('.env'), override=True)

# Agent state
class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Utilities
SKIP_DIRS = {'.git', 'node_modules', '.venv', 'venv', '__pycache__', '.mypy_cache', '.ruff_cache', '.idea', '.next', 'dist', 'build'}
CODE_EXTS = {'.py', '.ts', '.tsx', '.js', '.jsx', '.json', '.yml', '.yaml', '.toml', '.md', '.go', '.rs', '.java', '.rb', '.php', '.c', '.cc', '.cpp', '.h', '.hpp'}


def resolve_path(file_path: str) -> Optional[str]:
    """Resolve a path; if not found, try to locate by basename under CWD (skipping bulky dirs)."""
    candidate = os.path.abspath(os.path.expanduser(file_path))
    if os.path.isfile(candidate):
        return candidate

    basename = os.path.basename(file_path)
    cwd = os.getcwd()
    for root, dirs, files in os.walk(cwd):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        if basename in files:
            return os.path.join(root, basename)
    return None

# Tools
class CalculatorToolInput(BaseModel):
    operation: Literal["add","subtract","multiply","divide"] = Field(description="The operation to perform ('add', 'subtract', 'multiply', 'divide').")
    a: Union[int, float] = Field(description="The first number.")
    b: Union[int, float] = Field(description="The second number.")

class CalculatorTool(BaseTool):
    name: str = 'maths_calculator'
    description: str = """Define a two-input calculator tool.

    Arg:
        operation (str): The operation to perform ('add', 'subtract', 'multiply', 'divide').
        a (float or int): The first number.
        b (float or int): The second number.
        
    Returns:
        result (float or int): the result of the operation
    Example
        Divide: result   = a / b
        Subtract: result = a - b"""
    

    def __init__(self, console: Optional[Console] = None, **kwargs: Dict):
        super().__init__(**kwargs)
        self._console = console or Console()

    def _run(self, operation: str, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        if operation == 'divide' and b == 0:
            return {"error": "Division by zero is not allowed."}

        # Perform calculation
        if operation == 'add':
            result = a + b
        elif operation == 'subtract':
            result = a - b
        elif operation == 'multiply':
            result = a * b
        elif operation == 'divide':
            result = a / b
        else: 
            result = "unknown operation"
        return result

# Tools
class FileReadToolInput(BaseModel):
    file_path: str = Field(description='The path to the file to read')

class FileReadTool(BaseTool):
    name: str = 'file_read'
    description: str = ('Reads the entire content of a file given its path.'
                        'Use this whenever the user asks what is in a file or requests the full contents.')
    
    args_schema: ClassVar[type[BaseModel]] = FileReadToolInput

    def __init__(self, console: Optional[Console] = None, **kwargs: Dict):
        super().__init__(**kwargs)
        self._console = console or Console()

    def _run(self, file_path: str) -> str:  # type: ignore[override]
        try:
            resolved = resolve_path(file_path)
            if not resolved:
                cwd = os.getcwd()
                msg = Text.assemble(
                    ('Error: file not found\n', 'bold red'),
                    ('Requested: ', 'bold'), f'{file_path}\n',
                    ('Working dir: ', 'bold'), f'{cwd}\n',
                    ('Hint: ', 'bold'), 'Pass an absolute path or ensure the file exists under the current project.\n',
                )
                self._console.print(msg)
                return f"Error reading file: Not found -> {file_path} (cwd={cwd})"

            with open(resolved, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(resolved, 'rb') as fb:  # type: ignore[arg-type]
                    return fb.read().decode('utf-8', errors='replace')
            except Exception as e2:
                return f'Error reading file: {e2}'
        except PermissionError:
            return f'Error reading file: Permission denied -> {file_path}'
        except Exception as e:
            return f'Error reading file: {e}'


class FileViewToolInput(BaseModel):
    file_path: str = Field(description='Path to file to view')
    offset: int = Field(0, description='Zero-based line offset to start from')
    limit: int = Field(2000, description='Maximum number of lines to return')
    truncate_chars: int = Field(2000, description='Max characters per line before truncation')

class FileViewTool(BaseTool):
    name: str = 'file_view'
    description: str = (
        'Reads a slice of a file with line numbers. Accepts {"file_path", "offset", "limit", "truncate_chars"}.\n'
        'Lines longer than truncate_chars are truncated; each line is annotated with its number.'
    )
    args_schema: ClassVar[type[BaseModel]] = FileViewToolInput

    def __init__(self, console: Optional[Console] = None, **kwargs: Dict):
        super().__init__(**kwargs)
        self._console = console or Console()

    def _run(self, file_path: str, offset: int = 0, limit: int = 2000, truncate_chars: int = 2000) -> str:  # type: ignore[override]
        resolved = resolve_path(file_path)
        if not resolved:
            return f'Error file_view: Not found -> {file_path}'
        try:
            with open(resolved, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.read().splitlines()
            start = max(0, offset)
            end = min(len(lines), start + max(0, limit))
            out_lines = []
            for idx in range(start, end):
                ln = lines[idx]
                if len(ln) > truncate_chars:
                    ln = ln[:truncate_chars] + '…'
                out_lines.append(f"{idx+1:>6} | {ln}")
            header = f"file_view -> {resolved}\nShowing lines {start+1}–{end} of {len(lines)} (truncate>{truncate_chars} chars)"
            return header + "\n" + "\n".join(out_lines)
        except Exception as e:
            return f'Error file_view: {e}'

class FolderCreateToolInput(BaseModel):
    path: str = Field(description="Full path (absolute or relative) to create the folder.")

class FolderCreateTool(BaseTool):
    name: str = "folder_create"
    description: str = "Creates a new folder (directory) at the given path."
    args_schema: ClassVar[type[BaseModel]] = FolderCreateToolInput

    def _run(self, path: str) -> str:  # type: ignore[override]
        try:
            abs_path = os.path.abspath(os.path.expanduser(path))
            if os.path.exists(abs_path):
                return json.dumps({"folder": abs_path, "created": False, "message": "Folder already exists."})
            os.makedirs(abs_path, exist_ok=True)
            return json.dumps({"folder": abs_path, "created": "Folder created at given path."})
        except Exception as e:
            return f"Error creating folder: {e}"

class FileCreateToolInput(BaseModel):
    path: str = Field(description="Full path (absolute or relative) to create the file.")
    content: Optional[str] = Field("", description="Optional initial content to write in the file.")


class FileCreateTool(BaseTool):
    name: str = "file_create"
    description: str = "Creates a new file at the given path. Optionally writes initial content."
    args_schema: ClassVar[type[BaseModel]] = FileCreateToolInput

    def _run(self, path: str, content: str = "") -> str:  # type: ignore[override]
        try:
            abs_path = os.path.abspath(os.path.expanduser(path))

            # Ensure parent directory exists
            parent_dir = os.path.dirname(abs_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            if os.path.exists(abs_path):
                return json.dumps({
                    "file": abs_path,
                    "created": False,
                    "message": "File already exists."
                })

            # Actually create and write the file
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content or "")

            return json.dumps({
                "file": abs_path,
                "created": True,
                "message": "File created successfully."
            })
        except Exception as e:
            return f"Error creating file: {e}"
        
class FileEditToolInput(BaseModel):
    file_path: str
    old_string: str
    new_string: str
    count: Optional[int] = Field(None, description='Max replacements; None means replace all')
    fuzzy: bool = Field(False, description='Enable fuzzy matching of old_string against file content')
    threshold: float = Field(0.8, ge=0.0, le=1.0, description='Fuzzy match threshold (0..1) when fuzzy=true')
    preview_context: int = Field(3, ge=0, description='Unified diff context lines in preview')

class FileEditTool(BaseTool):
    name: str = 'file_edit'
    description: str = (
        'Replace occurrences of old_string with new_string inside a file.\n'
        'Shows a unified diff preview and asks for confirmation.\n'
        'Supports fuzzy matching with a configurable threshold.'
    )
    args_schema: ClassVar[type[BaseModel]] = FileEditToolInput

    def __init__(self, console: Optional[Console] = None, **kwargs: Dict):
        super().__init__(**kwargs)
        self._console = console or Console()

    def _make_diff(self, before: str, after: str, path: str, ctx: int) -> str:
        before_lines = before.splitlines(keepends=True)
        after_lines = after.splitlines(keepends=True)
        diff = difflib.unified_diff(before_lines, after_lines, fromfile=f"a/{path}", tofile=f"b/{path}", n=ctx)
        return ''.join(diff)[:20000]

    def _apply_exact(self, content: str, old: str, new: str, count: Optional[int]) -> tuple[str, int]:
        pattern = re.escape(old)
        max_count = 0 if count is None else max(0, count)
        return re.subn(pattern, new, content, count=max_count)

    def _apply_fuzzy(self, content: str, old: str, new: str, threshold: float, count: Optional[int]) -> tuple[str, int]:
        lines = content.splitlines(keepends=False)
        replaced = 0
        max_repl = float('inf') if count in (None, 0) else max(0, count)
        out_lines: list[str] = []
        window = len(old)
        for line in lines:
            if replaced >= max_repl or window == 0:
                out_lines.append(line)
                continue
            best_i = None
            best_score = 0.0
            # If line shorter than window, compare full line
            if len(line) < window and len(line) > 0:
                best_i = 0
                best_score = difflib.SequenceMatcher(None, line, old).ratio()
            else:
                for i in range(0, max(1, len(line) - window + 1)):
                    chunk = line[i:i+window]
                    score = difflib.SequenceMatcher(None, chunk, old).ratio()
                    if score > best_score:
                        best_score = score
                        best_i = i
            if best_i is not None and best_score >= threshold:
                new_line = line[:best_i] + new + line[best_i+window:]
                out_lines.append(new_line)
                replaced += 1
            else:
                out_lines.append(line)
        return ('\n'.join(out_lines), replaced)

    def _run(self, file_path: str, old_string: str, new_string: str, count: Optional[int] = None, fuzzy: bool = False, threshold: float = 0.8, preview_context: int = 3) -> str:  # type: ignore[override]
        resolved = resolve_path(file_path)
        if not resolved:
            return f'Error file_edit: Not found -> {file_path}'
        try:
            with open(resolved, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            # Draft change
            if fuzzy:
                new_content, n_done = self._apply_fuzzy(content, old_string, new_string, threshold, count)
            else:
                new_content, n_done = self._apply_exact(content, old_string, new_string, count)

            if n_done == 0:
                return json.dumps({'file': resolved, 'replacements': 0, 'message': 'No matches found (try fuzzy=true or lower threshold).'})

            # Show diff preview, ask user to accept/edit/cancel
            diff_text = self._make_diff(content, new_content, resolved, preview_context)
            self._console.print(Panel.fit(Syntax(diff_text or '(no visible diff)', 'diff', line_numbers=False), title='Proposed Changes'))
            prompt = 'Accept changes? [y]es / [e]dit params / [n]o: '
            ans = self._console.input(Text(prompt, style='bold yellow')).strip().lower()
            if ans in {'n', 'no'}:
                return json.dumps({'file': resolved, 'cancelled': "User requested to cancel the modification. Ask him for any further change", 'replacements': 0})
            if ans in {'e', 'edit'}:
                # lightweight param tweak loop
                self._console.print('[bold]Edit mode[/bold]: press Enter to keep current value.')
                new_old = self._console.input(f'old_string [{old_string}]: ').strip() or old_string
                new_new = self._console.input(f'new_string [{new_string}]: ').strip() or new_string
                if fuzzy:
                    try:
                        th_in = self._console.input(f'threshold [{threshold} 0..1]: ').strip()
                        threshold = float(th_in) if th_in else threshold
                    except ValueError:
                        pass
                cnt_in = self._console.input(f'count [{count if count is not None else "all"}]: ').strip()
                count2 = None if cnt_in == '' or cnt_in.lower() == 'all' else max(0, int(cnt_in))
                # Recompute and show preview again
                if fuzzy:
                    new_content, n_done = self._apply_fuzzy(content, new_old, new_new, threshold, count2)
                else:
                    new_content, n_done = self._apply_exact(content, new_old, new_new, count2)
                diff_text = self._make_diff(content, new_content, resolved, preview_context)
                self._console.print(Panel.fit(Syntax(diff_text or '(no visible diff)', 'diff', line_numbers=False), title='Proposed Changes (Edited)'))
                ans2 = self._console.input(Text('Accept edited changes? [y/N]: ', style='bold yellow')).strip().lower()
                if ans2 not in {'y', 'yes'}:
                    return json.dumps({'file': resolved, 'cancelled': "User requested to cancel the modification. Ask him for any further change", 'replacements': 0})

            # Commit
            with open(resolved, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return json.dumps({'file': resolved, 'replacements': n_done})
        except Exception as e:
            return f'Error file_edit: {e}'


class DispatchAgentInput(BaseModel):
    prompt: str = Field(description='Instruction describing what to search for (e.g., Find references to "encryptPassword" in userAuth.js)')
    threshold: float = Field(0.7, ge=0.0, le=1.0, description='Fuzzy threshold (0..1) for matching the reference token')
    max_results: int = Field(200, gt=0, description='Maximum number of matches to return')

class DispatchAgentTool(BaseTool):
    name: str = 'dispatch_agent'
    description: str = (
        'Specialized fuzzy search across the repo. Understands prompts like: Find references to "encryptPassword" in userAuth.js.\n'
        'Uses fuzzy matching (difflib) and returns file, line, snippet, and a score so the model/user can judge relevance.'
    )
    args_schema: ClassVar[type[BaseModel]] = DispatchAgentInput

    def _token_score(self, needle: str, text: str) -> float:
        # quick best-substring score of length len(needle)
        n = len(needle)
        if n == 0:
            return 0.0
        if len(text) <= n:
            return difflib.SequenceMatcher(None, needle, text).ratio()
        best = 0.0
        for i in range(0, len(text) - n + 1):
            s = difflib.SequenceMatcher(None, needle, text[i:i+n]).ratio()
            if s > best:
                best = s
        return best

    def _run(self, prompt: str, threshold: float = 0.7, max_results: int = 200) -> str:  # type: ignore[override]
        # Heuristic: extract quoted token and optional filename after "in "
        needle = None
        filename_filter = None
        m = re.search(r'\"([^\"]+)\"|\'([^\']+)\'', prompt)
        if m:
            needle = m.group(1) or m.group(2)
        m2 = re.search(r'\bin\s+([\w\./\\-]+)', prompt)
        if m2:
            filename_filter = os.path.basename(m2.group(1))
        if not needle:
            toks = re.findall(r'[A-Za-z0-9_\.]+', prompt)
            needle = toks[-1] if toks else ''
        if not needle:
            return json.dumps({'matches': [], 'message': 'No search token found.'})

        results: List[Dict[str, Any]] = []
        cwd = os.getcwd()
        for root, dirs, files in os.walk(cwd):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fn in files:
                if filename_filter and os.path.basename(fn) != filename_filter:
                    continue
                ext = os.path.splitext(fn)[1]
                if ext and ext not in CODE_EXTS:
                    continue
                path = os.path.join(root, fn)
                try:
                    with open(path, 'r', encoding='utf-8', errors='replace') as f:
                        for i, line in enumerate(f, start=1):
                            score = self._token_score(needle, line)
                            if score >= threshold:
                                results.append({'file': path, 'line': i, 'text': line.rstrip('\n'), 'score': round(score, 3)})
                except Exception:
                    continue

        # Sort by score desc, cap
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:max_results]
        # add simple decision hint for UI/LLM
        for r in results:
            s = r['score']
            r['decision_hint'] = 'strong' if s >= 0.9 else ('medium' if s >= 0.8 else 'weak')
        return json.dumps({'query': needle, 'filter': filename_filter, 'threshold': threshold, 'matches': results}, ensure_ascii=False)

# Agent
class SimpleAgent:
    def __init__(self):
        self.console = Console()
        self.console.print(Panel.fit('[green]Welcome to [/green][yellow]Deep Agentic [/yellow][red]Coder[/red]'), style='bold blue')

        # Register tools (NOTE: pass console to tools that prompt for confirmation)
        self.tools: List[BaseTool] = [
            FileReadTool(console=self.console),
            FileViewTool(console=self.console),
            FileEditTool(console=self.console),
            DispatchAgentTool(),
            FolderCreateTool(),
            FileCreateTool(),
            CalculatorTool()
        ]

        # Initialize the OpenAI chat model with GPT-4.1-mini, setting temperature to 0 for deterministic output, no max tokens limit, and retry policy
        self.model = init_chat_model(model='openai:gpt-4.1-mini', temperature=0, max_tokens=None, timeout=None, max_retries=2)
        self.model = self.model.bind_tools(self.tools)

        workflow = StateGraph(AgentState)
        workflow.add_node('user_input', self._get_user_input)
        workflow.add_node('model_response', self._get_model_response)
        workflow.add_node('tool_use', self._get_tool_use)

        workflow.set_entry_point('user_input')
        workflow.add_edge('user_input', 'model_response')
        workflow.add_conditional_edges('model_response', self._check_tool_use, {
            'tool_use': 'tool_use',
            'user_input': 'user_input'
        })
        workflow.add_edge('tool_use', 'model_response')
        self.agent = workflow.compile()


    # Workflow nodes
    async def _get_user_input(self, state: AgentState) -> AgentState:
        self.console.print('[bold yellow]User Input[/bold yellow]')
        user_input = self.console.input('[bold blue]>[/bold blue] ')
        return {'messages': [HumanMessage(content=user_input)]}

    async def _get_model_response(self, state: AgentState) -> AgentState:
        sys = SystemMessage(content=[{
            'type': 'text',
            'text': (
                prompt.SYSTEM_MESSAGE
            ),
            'cache_control': {'type': 'ephemeral'}
        }])
        working_dir_msg = HumanMessage(content=f'Working directory: {os.getcwd()}')
        
        messages = [sys, working_dir_msg] + state.messages

        self.console.print('[bold green]Model Response[/bold green]')
        response = await self.model.ainvoke(messages)

        if not getattr(response, 'tool_calls', None):
            md = Markdown(response.content)
            self.console.print(Panel.fit(md, title='Assistant'), style='green')
        else:
            try:
                name = response.tool_calls[0].get('name')
                self.console.print(Panel.fit(f'[bold red]Function calling [/bold red] {name}', title='Tool Use'), style='red')
            except Exception:
                pass

        return {'messages': [response]}

    async def _get_tool_use(self, state: AgentState) -> AgentState:
        tools_by_name = {tool.name: tool for tool in self.tools}
        last = state.messages[-1]
        responses: List[BaseMessage] = []

        for call in getattr(last, 'tool_calls', []) or []:
            tool_name = call.get('name')
            args = call.get('args', {})
            tool = tools_by_name.get(tool_name)
           
            if not tool:
                payload = json.dumps({'error': f'Unknown tool {tool_name}', 'args': args})
                responses.append(ToolMessage(content=payload, name=tool_name or 'unknown', tool_call_id=call.get('id')))
                continue
            try:
                if hasattr(tool, '_arun'):
                    result = await tool._arun(**args)  # type: ignore[misc]
                elif hasattr(tool, '_run'):
                    result = tool._run(**args)  # type: ignore[misc]
                    
            except TypeError as e:
                result = f'Tool invocation error for {tool_name}: {e}\nArgs: {args}'
            except Exception as e:
                result = f'Unexpected error during {tool_name}: {e}'

            responses.append(ToolMessage(content=str(result), name=tool_name, tool_call_id=call.get('id')))

        return {'messages': responses}

    async def _check_tool_use(self, state: AgentState) -> str:
        if getattr(state.messages[-1], 'tool_calls', None):
            return 'tool_use'
        else:
            return 'user_input'

    async def run(self) -> Union[List, Dict, str]:
        return self.console.print(await self.agent.ainvoke({'messages': ''}, config = {"recursion_limit": 50}))


if __name__ == '__main__':
    import asyncio
    agent = SimpleAgent()
    asyncio.run(agent.run())
