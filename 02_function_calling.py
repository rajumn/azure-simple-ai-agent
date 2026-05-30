"""
Azure AI Agents POC — Module 2: Function Calling
=================================================
Demonstrates:
  - Defining custom Python functions as agent tools
  - Registering them with JSON schema (OpenAI-compatible format)
  - Handling tool_call events and executing the functions locally
  - Submitting tool outputs back to the Run

The agent decides WHEN to call your functions; your code executes them.
"""

import json
import os
import random
from datetime import datetime
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    FunctionTool,
    RequiredFunctionToolCall,
    SubmitToolOutputsAction,
    ToolOutput,
    MessageRole,
)
from azure.identity import DefaultAzureCredential

load_dotenv()

client = AIProjectClient.from_connection_string(
    conn_str=os.environ["AZURE_AI_PROJECT_CONNECTION_STRING"],
    credential=DefaultAzureCredential(),
)

MODEL = os.getenv("AZURE_AI_AGENT_MODEL", "gpt-4o")


# ── Define your custom functions ─────────────────────────────────────────────

def get_current_weather(city: str, unit: str = "celsius") -> str:
    """Simulates a weather API call."""
    temp = round(random.uniform(15, 35), 1)
    return json.dumps({
        "city": city,
        "temperature": temp,
        "unit": unit,
        "condition": random.choice(["sunny", "cloudy", "partly cloudy", "rainy"]),
        "humidity": f"{random.randint(40, 90)}%",
        "fetched_at": datetime.utcnow().isoformat(),
    })


def get_stock_price(ticker: str) -> str:
    """Simulates a stock price lookup."""
    prices = {"MSFT": 415.23, "AAPL": 192.75, "GOOGL": 173.11, "AMZN": 181.90}
    price = prices.get(ticker.upper(), round(random.uniform(50, 500), 2))
    return json.dumps({
        "ticker": ticker.upper(),
        "price": price,
        "currency": "USD",
        "change_pct": round(random.uniform(-3, 3), 2),
        "as_of": datetime.utcnow().isoformat(),
    })


def send_notification(recipient: str, message: str, channel: str = "email") -> str:
    """Simulates sending a notification."""
    print(f"  📨 [NOTIFICATION] To: {recipient} via {channel}: {message}")
    return json.dumps({"status": "sent", "recipient": recipient, "channel": channel})


# ── JSON schema definitions (tell the agent what each function does) ─────────

FUNCTION_DEFINITIONS = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'London'"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["city"],
        },
    },
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker, e.g. 'MSFT'"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "send_notification",
        "description": "Send a notification to a recipient.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string"},
                "message": {"type": "string"},
                "channel": {"type": "string", "enum": ["email", "slack", "sms"]},
            },
            "required": ["recipient", "message"],
        },
    },
]

# Map function name → actual Python callable
FUNCTION_MAP = {
    "get_current_weather": get_current_weather,
    "get_stock_price": get_stock_price,
    "send_notification": send_notification,
}


def execute_tool_calls(tool_calls: list[RequiredFunctionToolCall]) -> list[ToolOutput]:
    """Execute all requested function calls and return outputs."""
    outputs = []
    for tc in tool_calls:
        fn_name = tc.function.name
        fn_args = json.loads(tc.function.arguments)

        print(f"  🔧 Calling function: {fn_name}({fn_args})")

        if fn_name in FUNCTION_MAP:
            result = FUNCTION_MAP[fn_name](**fn_args)
        else:
            result = json.dumps({"error": f"Unknown function: {fn_name}"})

        outputs.append(ToolOutput(tool_call_id=tc.id, output=result))
    return outputs


def demo_function_calling():
    """Agent decides when to call your custom Python functions."""
    print("\n=== Function Calling Demo ===\n")

    # 1. Create agent with FunctionTool definitions
    agent = client.agents.create_agent(
        model=MODEL,
        name="multi-tool-agent",
        instructions=(
            "You are a helpful assistant. Use the available tools to answer "
            "questions about weather and stocks. After gathering data, if the "
            "user asks to send a report, use send_notification."
        ),
        tools=[FunctionTool(functions=FUNCTION_DEFINITIONS)],
    )
    print(f"✓ Agent created: {agent.id}")

    thread = client.agents.create_thread()
    print(f"✓ Thread created: {thread.id}")

    # 2. Send a complex prompt that will trigger multiple function calls
    client.agents.create_message(
        thread_id=thread.id,
        role=MessageRole.USER,
        content=(
            "What's the weather in Tokyo and Paris right now? "
            "Also check the stock price for MSFT and AAPL. "
            "Then send a summary email to manager@company.com."
        ),
    )
    print("✓ User message added — starting run...\n")

    # 3. Create run WITHOUT auto-processing so we can handle tool calls manually
    run = client.agents.create_run(thread_id=thread.id, agent_id=agent.id)

    # 4. Poll loop — handle requires_action (function call requests)
    import time
    while run.status in ("queued", "in_progress", "requires_action"):
        time.sleep(1)
        run = client.agents.get_run(thread_id=thread.id, run_id=run.id)
        print(f"  Run status: {run.status}")

        if run.status == "requires_action":
            # Agent wants to call our functions
            submit_action: SubmitToolOutputsAction = run.required_action
            tool_calls = submit_action.submit_tool_outputs.tool_calls

            print(f"  Agent requested {len(tool_calls)} function call(s):")
            outputs = execute_tool_calls(tool_calls)

            # Submit results back to the agent
            run = client.agents.submit_tool_outputs_to_run(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=outputs,
            )

    print(f"\n✓ Run completed: status={run.status}")

    # 5. Print final response
    messages = client.agents.list_messages(thread_id=thread.id)
    for msg in messages.data:
        if msg.role == MessageRole.ASSISTANT:
            for item in msg.content:
                if hasattr(item, "text"):
                    print(f"\n[Agent Final Response]\n{item.text.value}")
            break  # most recent assistant message

    client.agents.delete_agent(agent.id)
    print("\n✓ Cleanup done")


if __name__ == "__main__":
    demo_function_calling()
