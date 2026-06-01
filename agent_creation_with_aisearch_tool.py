from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    AzureAISearchTool,
    AzureAISearchToolResource,
    AISearchIndexResource,
    PromptAgentDefinition,
    AzureAISearchQueryType,
)
from dotenv import load_dotenv
import os

load_dotenv()

project = AIProjectClient(
    endpoint=os.environ["AZURE_AI_PROJECT_CONNECTION_STRING"],
    credential=DefaultAzureCredential()
)

conn = project.connections.get(
    name="nameoftheaiseacrhconnectionInproject"
)

search_tool = AzureAISearchTool(
    azure_ai_search=AzureAISearchToolResource(
        indexes=[
            AISearchIndexResource(
                project_connection_id=conn.id,
                index_name="pdf-rag-index",
                query_type=AzureAISearchQueryType.SIMPLE,
            ),
        ]
    )
)

agent = project.agents.create_version(
    agent_name="search-agent",
    definition=PromptAgentDefinition(
        model=os.environ["AZURE_AI_AGENT_MODEL"],
        instructions="""
    Always must use Azure AI Search before answering.
    Cite sources.
    """,
        tools=[search_tool],
    ),
    description="Search agent with AI Search capabilities",
)
print("agent-", agent)