import os
from importlib.metadata import version
from dotenv import load_dotenv
from typing import Annotated, List, Literal, Union, Dict

from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool

from pydantic import BaseModel, Field

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

load_dotenv(os.path.join(".env"), override=True)

class TypeCheck(BaseModel):
    result: str = Field(description="The result of the model's response")

class FileReadToolInput(BaseModel):
    file_path: str = Field(description="The path to the file to read")

class FileReadTool(BaseTool):
    name: str = "file_read"
    description: str = "Reads the content of a file given its path."
    args_schema = FileReadToolInput

    def _run(self, file_path: str) -> str:
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except Exception as e:
            return f"Error reading file: {e}"

class SimpleAgent:
    def __init__(self):
        self.console = Console()
        self.console.print(Panel.fit("Hello, World!", style="bold green"))
        self.tools = [FileReadTool()]
        self.model = ChatOpenAI(model="gpt-5-chat-latest", temperature=0.2, max_tokens=None, timeout=None, max_retries=2)
        self.model.bind_tools(self.tools)

    async def run(self) -> str:
        result = TypeCheck(result=self.model.invoke("Hello, how are you?").content)
        return self.console.print(result.result, style="blue")
    
if __name__ == "__main__":
    import asyncio
    agent = SimpleAgent()
    asyncio.run(agent.run())