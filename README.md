# Azure AI Agents POC — Python

A hands-on POC covering all four pillars of Azure AI Agent Service.

## Prerequisites

1. **Azure subscription** with Azure AI Foundry project created
2. **Python 3.11+**
3. **Azure CLI** logged in (`az login`)

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — fill in your connection string and model deployment name
```

### Getting your Connection String

1. Go to [Azure AI Foundry](https://ai.azure.com)
2. Open your **project**
3. Click **Overview** → copy **"Project connection string"**

It looks like: `eastus.api.azureml.ms;11111111-...;my-rg;my-project`

---

## Module Overview

| File | Concept | Key APIs |
|------|---------|----------|
| `01_tool_execution.py` | Built-in tools (Code Interpreter, File Search) | `create_agent`, `create_and_process_run` |
| `02_function_calling.py` | Custom Python function calling | `FunctionTool`, `submit_tool_outputs_to_run` |
| `03_stateful_memory.py` | Thread-based persistent memory | `create_thread`, multi-turn `create_and_process_run` |
| `04_human_approval.py` | Human-in-the-loop approval gate | `requires_action`, `cancel_run` |

---

## Core Concepts

### 1. Tool Execution
Agents can use **built-in tools** — no extra code needed:
- **Code Interpreter** — runs Python in a sandboxed environment
- **File Search** — RAG over uploaded documents (vector store)
- **Bing Search** — live web search
- **Azure AI Search** — search your own Azure index
- **OpenAPI** — call any REST API from a spec

### 2. Function Calling
Define Python functions + their JSON schemas.  
The agent decides **when** to call them; your code **executes** them.

```
User prompt → Agent reasoning → "I need to call get_weather(city='Tokyo')"
→ Run status: requires_action → Your code executes → Submit output → Agent responds
```

### 3. Stateful Memory — Thread Architecture

```
Agent (stateless) + Thread (stateful) = Stateful Conversation

Thread
  ├── Message: User "Hello"
  ├── Message: Assistant "Hi! How can I help?"
  ├── Message: User "What did I say first?"
  └── Message: Assistant "You said 'Hello'"  ← remembers context
```

The **Thread** is persisted in Azure. You can resume conversations
by saving and reusing the `thread_id` across sessions.

### 4. Human Approval Flow (HITL)

```
Run lifecycle:
queued → in_progress → requires_action → [human reviews]
                                          ↓              ↓
                                      approved         rejected
                                          ↓              ↓
                               submit_tool_outputs   cancel_run
                                          ↓
                                      completed
```

---

## Run the Demos

```bash
# Module 1: Built-in tools
python 01_tool_execution.py

# Module 2: Function calling
python 02_function_calling.py

# Module 3: Stateful memory
python 03_stateful_memory.py

# Module 4: Human approval
python 04_human_approval.py
```

---

## Key SDK Classes

| Class | Purpose |
|-------|---------|
| `AIProjectClient` | Entry point — wraps the Azure AI Foundry project |
| `AIProjectClient.agents` | All agent/thread/run operations |
| `CodeInterpreterTool` | Enable code execution tool |
| `FileSearchTool` | Enable RAG over uploaded files |
| `FunctionTool` | Register custom Python functions |
| `MessageRole.USER / ASSISTANT` | Message sender roles |
| `SubmitToolOutputsAction` | The action to take when run is `requires_action` |
| `ToolOutput` | Wraps your function's return value for submission |

---

## Authentication

Uses `DefaultAzureCredential` — works with:
- `az login` (local development)  
- Managed Identity (Azure VMs, Azure Functions, AKS)
- Environment variables (`AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`)

---

## Further Reading

- [Azure AI Agents Overview](https://learn.microsoft.com/azure/ai-services/agents/overview)
- [Azure AI Projects SDK](https://learn.microsoft.com/python/api/overview/azure/ai-projects-readme)
- [Quickstart](https://learn.microsoft.com/azure/ai-services/agents/quickstart)
- [Tool reference](https://learn.microsoft.com/azure/ai-services/agents/tools/overview)
