"""
Azure AI Agents POC — Module 4: Human Approval Flows
=====================================================
Demonstrates:
  - requires_action run status as the approval gate
  - Pausing execution to get human confirmation
  - Approving or rejecting specific tool calls
  - Cancelling a run that awaits human input
  - Building a full human-in-the-loop (HITL) workflow

Pattern:
  Agent requests fn call → Run enters requires_action → 
  Human reviews → Approve (submit outputs) or Reject (cancel run)
"""

import json
import os
import time
from typing import Callable
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


# ── High-risk functions that require human approval ───────────────────────────

def delete_database_records(table: str, condition: str) -> str:
    """DANGER: Deletes records. Must require human approval before execution."""
    print(f"  ⚠️  [EXECUTING] DELETE FROM {table} WHERE {condition}")
    return json.dumps({"status": "deleted", "table": table, "condition": condition, "rows_affected": 42})


def send_mass_email(recipients_count: int, subject: str, body: str) -> str:
    """Sends email to many people. Should require human approval."""
    print(f"  📧 [EXECUTING] Sending '{subject}' to {recipients_count} recipients")
    return json.dumps({"status": "sent", "count": recipients_count, "subject": subject})


def deploy_to_production(service: str, version: str) -> str:
    """Production deployment — always needs human sign-off."""
    print(f"  🚀 [EXECUTING] Deploying {service} v{version} to PRODUCTION")
    return json.dumps({"status": "deployed", "service": service, "version": version})


def get_system_health() -> str:
    """Safe read-only operation — no approval needed."""
    return json.dumps({"cpu": "45%", "memory": "62%", "status": "healthy", "services_up": 12})


FUNCTION_DEFINITIONS = [
    {
        "name": "delete_database_records",
        "description": "Delete records from a database table matching a condition.",
        "parameters": {
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "condition": {"type": "string", "description": "SQL WHERE clause condition"},
            },
            "required": ["table", "condition"],
        },
    },
    {
        "name": "send_mass_email",
        "description": "Send a bulk email campaign to many recipients.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipients_count": {"type": "integer"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["recipients_count", "subject", "body"],
        },
    },
    {
        "name": "deploy_to_production",
        "description": "Deploy a service version to the production environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "version": {"type": "string"},
            },
            "required": ["service", "version"],
        },
    },
    {
        "name": "get_system_health",
        "description": "Check the current health of all systems.",
        "parameters": {"type": "object", "properties": {}},
    },
]

FUNCTION_MAP = {
    "delete_database_records": delete_database_records,
    "send_mass_email": send_mass_email,
    "deploy_to_production": deploy_to_production,
    "get_system_health": get_system_health,
}

# Functions that require human approval before execution
HIGH_RISK_FUNCTIONS = {"delete_database_records", "send_mass_email", "deploy_to_production"}


def human_approval_gate(tool_call: RequiredFunctionToolCall) -> bool:
    """
    Simulate human review of a requested tool call.
    In production, this would trigger:
      - A Slack message / email to an approver
      - A UI approval screen
      - An audit log entry
    Returns True to approve, False to reject.
    """
    fn_name = tool_call.function.name
    fn_args = json.loads(tool_call.function.arguments)

    print(f"\n  ┌── HUMAN APPROVAL REQUIRED ──────────────────────────┐")
    print(f"  │ Function : {fn_name}")
    print(f"  │ Arguments: {json.dumps(fn_args, indent=4).replace(chr(10), chr(10)+'  │            ')}")
    print(f"  │ Risk Level: HIGH ⚠️")
    print(f"  └─────────────────────────────────────────────────────┘")

    # In a real app: user_input = input("Approve? (y/n): ")
    # Here we auto-approve for the demo — change to False to test rejection
    user_decision = True  # Simulates human clicking "Approve"

    if user_decision:
        print("  ✅ Human approved")
    else:
        print("  ❌ Human rejected")

    return user_decision


def run_with_human_approval(
    agent_id: str,
    thread_id: str,
    approval_fn: Callable[[RequiredFunctionToolCall], bool],
):
    """
    Run loop that pauses on requires_action for human approval.
    Approved calls are executed; rejected calls cancel the entire run.
    """
    run = client.agents.create_run(thread_id=thread_id, agent_id=agent_id)
    print(f"\n  Run started: {run.id}")

    while run.status in ("queued", "in_progress", "requires_action"):
        time.sleep(1)
        run = client.agents.get_run(thread_id=thread_id, run_id=run.id)
        print(f"  Status: {run.status}")

        if run.status == "requires_action":
            action: SubmitToolOutputsAction = run.required_action
            tool_calls = action.submit_tool_outputs.tool_calls

            tool_outputs = []
            all_approved = True

            for tc in tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)

                if fn_name in HIGH_RISK_FUNCTIONS:
                    # Gate: ask human
                    approved = approval_fn(tc)
                    if not approved:
                        print(f"\n  🛑 Rejected by human. Cancelling run...")
                        client.agents.cancel_run(thread_id=thread_id, run_id=run.id)
                        all_approved = False
                        break
                else:
                    # Safe functions execute without approval
                    print(f"  ✓ Auto-executing safe function: {fn_name}")

                # Execute the approved or safe function
                result = FUNCTION_MAP[fn_name](**fn_args)
                tool_outputs.append(ToolOutput(tool_call_id=tc.id, output=result))

            if all_approved and tool_outputs:
                run = client.agents.submit_tool_outputs_to_run(
                    thread_id=thread_id,
                    run_id=run.id,
                    tool_outputs=tool_outputs,
                )

    return run


def demo_human_approval():
    """Full HITL workflow: agent requests risky actions, human approves."""
    print("\n=== Human Approval Flow Demo ===\n")

    agent = client.agents.create_agent(
        model=MODEL,
        name="ops-agent",
        instructions=(
            "You are an operations agent. When given tasks, use the available "
            "functions. Always check system health first, then proceed with requested operations."
        ),
        tools=[FunctionTool(functions=FUNCTION_DEFINITIONS)],
    )
    print(f"✓ Agent created: {agent.id}")

    thread = client.agents.create_thread()
    client.agents.create_message(
        thread_id=thread.id,
        role=MessageRole.USER,
        content=(
            "First check system health. Then delete all records from the 'temp_logs' table "
            "where created_at < '2024-01-01'. Finally, deploy 'payment-service' v2.1.0 to production."
        ),
    )
    print("✓ User message added — launching run with approval gate...\n")

    final_run = run_with_human_approval(
        agent_id=agent.id,
        thread_id=thread.id,
        approval_fn=human_approval_gate,
    )

    print(f"\n✓ Final run status: {final_run.status}")

    # Print agent's final response
    messages = client.agents.list_messages(thread_id=thread.id)
    for msg in messages.data:
        if msg.role == MessageRole.ASSISTANT:
            for item in msg.content:
                if hasattr(item, "text"):
                    print(f"\n[Agent Response]\n{item.text.value}")
            break

    client.agents.delete_agent(agent.id)
    print("\n✓ Cleanup done")


def demo_run_steps_audit():
    """Inspect Run Steps to see exactly what the agent did (audit trail)."""
    print("\n=== Run Steps Audit Trail Demo ===\n")

    agent = client.agents.create_agent(
        model=MODEL,
        name="audit-demo-agent",
        instructions="Check system health and report findings.",
        tools=[FunctionTool(functions=[FUNCTION_DEFINITIONS[3]])],  # get_system_health only
    )

    thread = client.agents.create_thread()
    client.agents.create_message(
        thread_id=thread.id,
        role=MessageRole.USER,
        content="Check system health and summarise it.",
    )

    run = client.agents.create_and_process_run(
        thread_id=thread.id, agent_id=agent.id
    )
    print(f"✓ Run completed: {run.status}")

    # Fetch run steps — complete audit trail of every action
    run_steps = client.agents.list_run_steps(thread_id=thread.id, run_id=run.id)
    print(f"\n📋 Audit Trail ({len(run_steps.data)} steps):")
    for i, step in enumerate(run_steps.data, 1):
        print(f"  Step {i}: type={step.type}, status={step.status}")
        if hasattr(step.step_details, "tool_calls"):
            for tc in step.step_details.tool_calls:
                print(f"    → Tool called: {tc.function.name}({tc.function.arguments})")

    client.agents.delete_agent(agent.id)
    print("\n✓ Cleanup done")


if __name__ == "__main__":
    demo_human_approval()
    # demo_run_steps_audit()  # Uncomment for audit trail demo
