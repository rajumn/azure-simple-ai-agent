"""
Azure AI Agents POC — Module 3: Stateful Memory
================================================
Demonstrates:
  - Threads as persistent conversation containers
  - Multi-turn conversations that remember context
  - Adding metadata to threads and messages
  - Listing / resuming existing threads (thread persistence)
  - Using thread_id to resume a previous session

Key insight: The Thread is the memory. Azure persists all messages.
The agent always sees the full conversation history on each Run.
"""

import json
import os
import time
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MessageRole
from azure.identity import DefaultAzureCredential

load_dotenv()

client = AIProjectClient.from_connection_string(
    conn_str=os.environ["AZURE_AI_PROJECT_CONNECTION_STRING"],
    credential=DefaultAzureCredential(),
)

MODEL = os.getenv("AZURE_AI_AGENT_MODEL", "gpt-4o")

# File to persist thread_id between Python runs (simulates a real app session store)
SESSION_FILE = ".agent_session.json"


def save_session(agent_id: str, thread_id: str):
    with open(SESSION_FILE, "w") as f:
        json.dump({"agent_id": agent_id, "thread_id": thread_id}, f)
    print(f"  💾 Session saved → {SESSION_FILE}")


def load_session() -> dict | None:
    try:
        with open(SESSION_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def chat_turn(agent_id: str, thread_id: str, user_message: str) -> str:
    """Send one user message and return the agent's response."""
    # Add user message to the thread
    client.agents.create_message(
        thread_id=thread_id,
        role=MessageRole.USER,
        content=user_message,
    )

    # Start a run (agent reads ALL messages in thread, including history)
    run = client.agents.create_and_process_run(
        thread_id=thread_id,
        agent_id=agent_id,
    )

    if run.status != "completed":
        return f"[Run ended with status: {run.status}]"

    # Get the most recent assistant message
    messages = client.agents.list_messages(thread_id=thread_id)
    for msg in messages.data:
        if msg.role == MessageRole.ASSISTANT:
            for item in msg.content:
                if hasattr(item, "text"):
                    return item.text.value
    return "[No response]"


def demo_new_conversation():
    """Start a fresh multi-turn conversation and persist the session."""
    print("\n=== New Stateful Conversation ===\n")

    agent = client.agents.create_agent(
        model=MODEL,
        name="stateful-tutor",
        instructions=(
            "You are a programming tutor. Keep track of what the student has "
            "learned so far. Reference earlier parts of the conversation when relevant. "
            "Be concise."
        ),
    )
    thread = client.agents.create_thread()
    save_session(agent.id, thread.id)

    print(f"Agent ID : {agent.id}")
    print(f"Thread ID: {thread.id}  (this is the stateful memory)\n")

    # Turn 1
    q1 = "I'm learning Python. What should I start with?"
    print(f"User: {q1}")
    r1 = chat_turn(agent.id, thread.id, q1)
    print(f"Agent: {r1}\n")

    # Turn 2 — agent should remember context from turn 1
    q2 = "Great, I understand variables now. What's next after that?"
    print(f"User: {q2}")
    r2 = chat_turn(agent.id, thread.id, q2)
    print(f"Agent: {r2}\n")

    # Turn 3 — even deeper follow-up
    q3 = "Can you give me a small exercise combining what I've learned so far?"
    print(f"User: {q3}")
    r3 = chat_turn(agent.id, thread.id, q3)
    print(f"Agent: {r3}\n")

    # Inspect full thread message history
    messages = client.agents.list_messages(thread_id=thread.id)
    print(f"✓ Thread now contains {len(messages.data)} messages (full history persisted)")

    return agent.id, thread.id


def demo_resume_conversation():
    """Resume a previous conversation using saved session (thread_id)."""
    print("\n=== Resuming Previous Conversation ===\n")

    session = load_session()
    if not session:
        print("No saved session found. Run demo_new_conversation() first.")
        return

    agent_id = session["agent_id"]
    thread_id = session["thread_id"]
    print(f"Resuming: agent={agent_id}, thread={thread_id}")

    # The agent will remember everything from the previous run
    q = "Remind me what topics we covered so far in our lesson."
    print(f"\nUser: {q}")
    response = chat_turn(agent_id, thread_id, q)
    print(f"Agent: {response}")

    # Inspect thread metadata
    thread = client.agents.get_thread(thread_id)
    messages = client.agents.list_messages(thread_id=thread_id)
    print(f"\n✓ Thread {thread.id} has {len(messages.data)} total messages")

    # Cleanup
    client.agents.delete_agent(agent_id)
    import os as _os
    if _os.path.exists(SESSION_FILE):
        _os.remove(SESSION_FILE)
    print("✓ Session and agent cleaned up")


def demo_thread_metadata():
    """Attach metadata to threads for filtering/retrieval in real apps."""
    print("\n=== Thread Metadata Demo ===\n")

    agent = client.agents.create_agent(
        model=MODEL,
        name="support-agent",
        instructions="You are a customer support agent.",
    )

    # Create thread with metadata tags (useful for multi-user apps)
    thread = client.agents.create_thread(
        metadata={
            "user_id": "user_12345",
            "session_type": "support",
            "product": "Azure AI Foundry",
            "priority": "high",
        }
    )
    print(f"✓ Thread created with metadata: {thread.metadata}")

    # Add a message with metadata
    msg = client.agents.create_message(
        thread_id=thread.id,
        role=MessageRole.USER,
        content="My agent deployment is failing. Can you help?",
        metadata={"source": "web-chat", "locale": "en-US"},
    )
    print(f"✓ Message created with metadata: {msg.metadata}")

    run = client.agents.create_and_process_run(
        thread_id=thread.id, agent_id=agent.id
    )
    print(f"✓ Run completed: {run.status}")

    client.agents.delete_agent(agent.id)
    print("✓ Cleanup done")


if __name__ == "__main__":
    agent_id, thread_id = demo_new_conversation()
    # To test resume, re-run the script and call demo_resume_conversation()
    # demo_resume_conversation()
    # demo_thread_metadata()
