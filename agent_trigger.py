import time
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ENDPOINT = os.environ["AZURE_AI_PROJECT_CONNECTION_STRING"]

# Existing agent id from your creation script
AGENT_ID = os.environ["AGENT_ID"]

project_client = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=DefaultAzureCredential(),
)

# Get OpenAI client for responses
openai_client = project_client.get_openai_client()

# Send message to agent and get response with streaming
stream_response = openai_client.responses.create(
    stream=True,
    tool_choice="required",
    input="What is the bill amount?",
    extra_body={
        "agent_reference": {
            "name": AGENT_ID,
            "type": "agent_reference"
        }
    }
)

print("=== RESPONSE ===\n")
# print(response)

# Process streamed response
for event in stream_response:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
    elif event.type == "response.text.done":
        print("\n")
    elif event.type == "response.output_item.done":
        if event.item.type == "message":
            item = event.item
            if item.content[-1].type == "output_text":
                text_content = item.content[-1]
                for annotation in text_content.annotations:
                    if annotation.type == "url_citation":
                        print(
                            f"\nSource: {annotation.url}"
                        )