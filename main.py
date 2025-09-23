import os
from importlib.metadata import version
from dotenv import load_dotenv
from typing import Annotated, List, Literal, Union, Dict
import time

from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from pydantic import BaseModel, Field

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

load_dotenv(os.path.join(".env"), override=True)

class FileReadToolInput(BaseModel):
    file_path: str = Field(description="The path to the file to read")

class FileReadTool(BaseTool):
    name: str = "file_read"
    description: str = "Reads the content of a file given its path."
    args_schema = FileReadToolInput

    def __init__(self, console: Console | None = None, **kwargs: Dict):
        super().__init__(**kwargs)
        self._console = console or Console()

    def _resolve_path(self, file_path: str) -> str | None:
        """
        Try to resolve to a real file:
        1) expanduser & abspath
        2) if missing, search recursively for the basename within CWD (skip bulky dirs)
        """
        # Normalize
        candidate = os.path.abspath(os.path.expanduser(file_path))
        if os.path.isfile(candidate):
            return candidate

        # Fallback search by filename
        basename = os.path.basename(file_path)
        cwd = os.getcwd()

        skip_dirs = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".ruff_cache"}
        for root, dirs, files in os.walk(cwd):
            # prune bulky dirs
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            if basename in files:
                return os.path.join(root, basename)

        return None

    def _read_with_loading(self, resolved_path: str) -> str:
        # Actual read
        with open(resolved_path, "r", encoding="utf-8") as f:
            return f.read()

    def _run(self, file_path: str) -> str:
        try:
            resolved = self._resolve_path(file_path)
            if not resolved:
                cwd = os.getcwd()
                msg = Text.assemble(
                    ("Error: file not found\n", "bold red"),
                    ("Requested: ", "bold"), f"{file_path}\n",
                    ("Working dir: ", "bold"), f"{cwd}\n",
                    ("Hint: ", "bold"), "Pass an absolute path or ensure the file exists under the current project.\n",
                )
                self._console.print(msg)
                return f"Error reading file: Not found -> {file_path} (cwd={cwd})"

            return self._read_with_loading(resolved)

        except UnicodeDecodeError as e:
            self._console.print(f"[bold red]Encoding error[/bold red]: {e}")
            # Re-read as binary then decode permissively
            try:
                with open(resolved, "rb") as fb:
                    return fb.read().decode("utf-8", errors="replace")
            except Exception as e2:
                return f"Error reading file: {e2}"

        except PermissionError as e:
            self._console.print(f"[bold red]Permission error[/bold red]: {e}")
            return f"Error reading file: Permission denied -> {file_path}"

        except Exception as e:
            self._console.print(f"[bold red]Unexpected error[/bold red]: {e}")
            return f"Error reading file: {e}"

class SimpleAgent:
    def __init__(self):
        self.console = Console()
        self.console.print(Panel.fit("[green]Welcome to [/green][yellow]Deep Agentic [/yellow][red]Coder[/red]"), style="bold blue")
        self.tools = [FileReadTool()]
        self.model = init_chat_model(model="openai:gpt-4.1-mini", temperature=0.2, max_tokens=None, timeout=None, max_retries=2)
        self.agent = create_react_agent(self.model, self.tools)

    async def run(self) -> str:
        result = self.agent.invoke({"messages": [("user", "read main.py file")]})
        return self.console.print(result["messages"][-1].content, style="blue")
    
if __name__ == "__main__":
    import asyncio
    agent = SimpleAgent()
    asyncio.run(agent.run())