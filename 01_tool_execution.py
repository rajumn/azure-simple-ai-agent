"""
Azure AI Agents POC — Module 1: Tool Execution
================================================
Demonstrates:
  - Creating an Agent with built-in tools (Code Interpreter, File Search)
  - Creating a Thread and sending messages
  - Polling a Run to completion
  - Reading the assistant's response

Docs: https://learn.microsoft.com/azure/ai-services/agents/overview
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    CodeInterpreterTool,
    FilePurpose,
    MessageRole,
)
from azure.identity import DefaultAzureCredential

load_dotenv()

# ── Client ──────────────────────────────────────────────────────────────────
client = AIProjectClient.from_connection_string(
    conn_str=os.environ["AZURE_AI_PROJECT_CONNECTION_STRING"],
    credential=DefaultAzureCredential(),
)

MODEL = os.getenv("AZURE_AI_AGENT_MODEL", "gpt-4o")


def demo_code_interpreter():
    """Agent uses Code Interpreter to run Python and analyze data."""
    print("\n=== Code Interpreter Tool Demo ===\n")

    # 1. Create agent with Code Interpreter tool
    agent = client.agents.create_agent(
        model=MODEL,
        name="data-analyst-agent",
        instructions=(
            "You are a data analyst. When asked to perform calculations "
            "or generate charts, write and execute Python code to do so."
        ),
        tools=[CodeInterpreterTool()],
    )
    print(f"✓ Agent created: {agent.id}")

    # 2. Create a stateful Thread (conversation session)
    thread = client.agents.create_thread()
    print(f"✓ Thread created: {thread.id}")

    # 3. Add a user message to the thread
    client.agents.create_message(
        thread_id=thread.id,
        role=MessageRole.USER,
        content=(
            "Calculate the compound interest on $10,000 at 7% annual rate "
            "over 20 years, compounding monthly. Show the year-by-year breakdown."
        ),
    )
    print("✓ User message added")

    # 4. Create a Run (triggers the agent to process the thread)
    run = client.agents.create_and_process_run(
        thread_id=thread.id,
        agent_id=agent.id,
    )
    print(f"✓ Run completed: status={run.status}")

    # 5. Fetch and print the assistant's response
    messages = client.agents.list_messages(thread_id=thread.id)
    for msg in messages.data:
        if msg.role == MessageRole.ASSISTANT:
            for content_item in msg.content:
                if hasattr(content_item, "text"):
                    print(f"\n[Agent Response]\n{content_item.text.value}")

    # 6. Cleanup (optional — keep agent/thread for stateful follow-ups)
    client.agents.delete_agent(agent.id)
    print(f"\n✓ Agent {agent.id} deleted")


def demo_file_search():
    """Agent uses File Search (RAG) to answer questions from uploaded documents."""
    print("\n=== File Search (RAG) Tool Demo ===\n")

    from azure.ai.projects.models import FileSearchTool, VectorStoreDataSource

    # 1. Upload a sample text file as a knowledge source
    sample_file = Path("sample_policy.txt")
    sample_file.write_text(
        "Company Policy v2.0\n"
        "Vacation: Employees get 20 days PTO per year.\n"
        "Remote Work: Hybrid (3 days office, 2 days remote) is the default.\n"
        "Benefits: Health, dental, and vision insurance provided from day 1.\n"
        "Parental Leave: 16 weeks fully paid for all parents.\n"
    )

    uploaded = client.agents.upload_file_and_poll(
        file_path=str(sample_file),
        purpose=FilePurpose.AGENTS,
    )
    print(f"✓ File uploaded: {uploaded.id}")
    sample_file.unlink()  # clean up local file

    # 2. Create a vector store from the uploaded file
    vector_store = client.agents.create_vector_store_and_poll(
        file_ids=[uploaded.id],
        name="company-policy-store",
    )
    print(f"✓ Vector store created: {vector_store.id}")

    # 3. Create agent with File Search pointing to our vector store
    agent = client.agents.create_agent(
        model=MODEL,
        name="policy-assistant",
        instructions="You answer HR policy questions using the provided documents.",
        tools=[FileSearchTool(vector_store_ids=[vector_store.id])],
    )
    print(f"✓ Agent created: {agent.id}")

    thread = client.agents.create_thread()
    client.agents.create_message(
        thread_id=thread.id,
        role=MessageRole.USER,
        content="How many vacation days do employees get, and what is the parental leave policy?",
    )

    run = client.agents.create_and_process_run(
        thread_id=thread.id,
        agent_id=agent.id,
    )
    print(f"✓ Run completed: status={run.status}")

    messages = client.agents.list_messages(thread_id=thread.id)
    for msg in messages.data:
        if msg.role == MessageRole.ASSISTANT:
            for item in msg.content:
                if hasattr(item, "text"):
                    print(f"\n[Agent Response]\n{item.text.value}")

    # Cleanup
    client.agents.delete_vector_store(vector_store.id)
    client.agents.delete_file(uploaded.id)
    client.agents.delete_agent(agent.id)
    print("\n✓ Resources cleaned up")


if __name__ == "__main__":
    demo_code_interpreter()
    # demo_file_search()   # Uncomment to test File Search too
