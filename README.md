# Deep-Agentic-Code-Agent
An autonomous AI-powered coding assistant that generates, debugs, and optimises code with advanced reasoning and contextual understanding.

## 🏗️ Architecture

### Core Agent
Built with [LangGraph](https://github.com/langchain-ai/langgraph) state graph for handling **input**, **tool calls**, and **model responses**.  
This design ensures structured reasoning and clean orchestration of tool usage.

### Model
Uses **`Open-AI Reasoning Model`** (configurable) via [LangChain OpenAI bindings](https://python.langchain.com/).  
Default configuration sets temperature to `0` for deterministic output, with retry and timeout handling.

### Rich Console UI
All interactions (diff previews, prompts, JSON outputs) are rendered with the [Rich](https://rich.readthedocs.io/) library, giving a clear, user-friendly terminal experience.

### Tools
- **`file_read`**: Read entire file content.  
- **`file_view`**: View file slices with numbered lines and truncated long lines.  
- **`file_edit`**: Perform targeted or fuzzy replacements with unified diff preview and confirmation.  
- **`dispatch_agent`**: Perform fuzzy semantic search across the repo with scoring and decision hints.  

---

## ⚙️ Running the Project with [uv](https://github.com/astral-sh/uv)

This project uses **uv** (a fast Python package manager) for reproducible environments.

### 1. Install `uv`
If you don’t have it yet:
```bash
pip install uv
uv sync

uv run main.py

```