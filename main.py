import os
from importlib.metadata import version
from dotenv import load_dotenv
from typing import Annotated, List, Literal, Union, Dict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages

import ast
import time

from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from langgraph.prebuilt.chat_agent_executor import AgentState, StateGraph

from pydantic import BaseModel, Field

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.text import Text
from rich.syntax import Syntax

load_dotenv(os.path.join(".env"), override=True)

class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]

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
        self.model = init_chat_model(model="openai:gpt-4.1-mini", temperature=0, max_tokens=None, timeout=None, max_retries=2)
        self.model = self.model.bind_tools(self.tools)
        # self.agent = create_react_agent(self.model, self.tools)

        workflow = StateGraph(AgentState)
        workflow.add_node("user_input", self._get_user_input)
        workflow.add_node("model_response", self._get_model_response)
        workflow.add_node("tool_use", self._get_tool_use)

        workflow.set_entry_point("user_input")
        workflow.add_edge("user_input", "model_response")
        workflow.add_conditional_edges("model_response", 
                                  self._check_tool_use,
                                  {
                                      "tool_use": "tool_use",
                                      "user_input": "user_input"
                                  })
        workflow.add_edge("tool_use", "model_response")

        self.agent = workflow.compile()

    async def _get_user_input(self, state: AgentState) -> AgentState:
        self.console.print("[bold yellow]User Input[/bold yellow]")
        user_input = self.console.input("[bold blue]>[/bold blue] ")
        return {"messages": [HumanMessage(content=user_input)]}
    
    async def _get_model_response(self, state: AgentState) -> AgentState:
        messages = [SystemMessage(content=[
            {
                "type" : "text",
                "text" : "You are a helpful AI assistant. Which has access to tools to work on local filesystem.",
                "cache_control": {
                    "type": "ephemeral"
                }
            }]),
        HumanMessage(content=f"Working directory: {os.getcwd()}")
        ] + state.messages
        self.console.print("[bold green]Model Response[/bold green]")
        response = await self.model.ainvoke(state.messages)

        try:
            if response.tool_calls[0]["type"]  == "tool_call":
                self.console.print(Panel.fit(f"[bold red]Function calling [/bold red] {response.tool_calls[0]['name']}", title="Tool Use"), style="red")
        except Exception as e:
            md = Markdown(response.content)
            self.console.print(Panel.fit(md, title="Assistant"), style="green")
        
        return {"messages": [response]}
    
    async def _get_tool_use(self, state: AgentState) -> AgentState:
        tools_by_name = {tool.name: tool for tool in self.tools}
        
        if state.messages[-1].tool_calls[0]:
            function = state.messages[-1].tool_calls[0]
            response = []
            tool_name = function["name"]
            tool_args = function.get('args')['file_path']
            tool = tools_by_name[tool_name]
            tool_result = tool._run(tool_args)

            self.console.print(Panel.fit(Syntax(tool_result, 'python', line_numbers=True), title="Tool Result"), style="magenta")
            response.append(ToolMessage(content=tool_result, name=tool_name, tool_call_id=state.messages[-1].tool_calls[0]['id']))

        return {"messages": response}
    
    async def _check_tool_use (self, state: AgentState) -> AgentState:
        if state.messages[-1].tool_calls:
            return "tool_use"
        else:
            return "user_input"
        
    async def run(self) -> Union[List, Dict, str]:
        return self.console.print(await self.agent.ainvoke({"messages": ""}))
    
if __name__ == "__main__":
    import asyncio
    agent = SimpleAgent()
    asyncio.run(agent.run())