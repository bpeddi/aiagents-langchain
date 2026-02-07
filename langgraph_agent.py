# LangGraph Agent with Web Search
# This script demonstrates a multi-node AI agent using LangGraph that can:
# - Decide if a query needs web search
# - Perform web searches using Tavily
# - Generate intelligent responses based on search results or knowledge

# pip install langgraph langchain langchain-openai langchain-community tavily-python python-dotenv

# ============ 1. Import Libraries ============
import os
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import operator
from langchain_core.runnables.graph_ascii import draw_ascii 
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Load environment variables (set OPENAI_API_KEY and TAVILY_API_KEY first)
load_dotenv()

# ============ 2. Define the Agent State ============
class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]  # Accumulates conversation history
    needs_search: bool  # Flag to determine if web search is needed

# ============ 3. Initialize Tools & LLM ============
# Check for required environment variables
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

if not anthropic_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables. Please set it in your .env file.")
if not tavily_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables. Please set it in your .env file.")

# Initialize LLM and search tool
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
search_tool = TavilySearchResults(max_results=3)

print("✅ LLM and search tool initialized successfully!")

llm = ChatAnthropic(
    model="claude-opus-4-6",  # or "claude-3-haiku-20240307"
    temperature=0,
    max_tokens=1024
)

# ============ 4. Define Graph Nodes ============
def should_search_node(state: AgentState) -> AgentState:
    """Decide if we need to search the web based on the query."""
    messages = [
        SystemMessage(
            content="Determine if this query requires current/real-time information "
            "or factual verification that might change over time. "
            "Respond ONLY with 'yes' or 'no'."
        ),
        HumanMessage(content=f"Query: {state['messages'][-1]['content']}")
    ]
    response = llm.invoke(messages)
    needs_search = "yes" in response.content.lower()
    return {"needs_search": needs_search}

def search_node(state: AgentState) -> AgentState:
    """Perform web search and add results to conversation."""
    query = state["messages"][-1]["content"]
    results = search_tool.invoke({"query": query})
    
    # Format search results
    formatted_results = "\n".join([
        f"Result {i+1}: {res['content']}" 
        for i, res in enumerate(results)
    ])
    
    # Add search results as an AI message
    return {
        "messages": [
            {
                "role": "assistant",
                "content": f"Search results for '{query}':\n{formatted_results}"
            }
        ]
    }

def respond_node(state: AgentState) -> AgentState:
    """Generate final response using conversation history."""
    # Extract the original user query (first message)
    user_query = None
    search_results = None
    
    for msg in state["messages"]:
        if msg["role"] == "user":
            user_query = msg["content"]
        elif msg["role"] == "assistant" and "Search results" in msg["content"]:
            search_results = msg["content"]
    
    # Build messages ensuring conversation ends with user message
    messages = [SystemMessage(content="You are a helpful assistant. Answer the user's question based on the provided information.")]
    
    if search_results:
        # Add search results as context
        messages.append(AIMessage(content=search_results))
    
    # Always end with user message
    messages.append(HumanMessage(content=user_query or state["messages"][-1]["content"]))
    print(f"🔍 Generating response with the following context: {messages}" )
    # for msg in messages:
    #     print(f"  {msg.role}: {msg.content}")
    response = llm.invoke(messages)
    return {
        "messages": [{"role": "assistant", "content": response.content}]
    }

# ============ 5. Define Conditional Routing ============
def route_after_decision(state: AgentState) -> str:
    """Route to search or direct response based on decision."""
    return "search" if state["needs_search"] else "respond"

# ============ 6. Build the Graph ============
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("decide", should_search_node)
workflow.add_node("search", search_node)
workflow.add_node("respond", respond_node)

# Set entry point
workflow.set_entry_point("decide")

# Add edges with conditional routing
workflow.add_conditional_edges(
    "decide",
    route_after_decision,
    {"search": "search", "respond": "respond"}
)
workflow.add_edge("search", "respond")
workflow.add_edge("respond", END)

# Compile the graph
app = workflow.compile()

# ============ 7. Visualize the Graph ============
print("📊 Agent Graph Structure:")
print("=" * 50)
print(app.get_graph().draw_ascii())

# ============ 8. Run the Agent ============
if __name__ == "__main__":
    # Example 1: Question needing search
    print("🔍 Example 1: Current events question")
    inputs = {"messages": [{"role": "user", "content": "What were the major AI announcements at CES 2026?"}], "needs_search": False}
    result = app.invoke(inputs)
    print("Answer:", result["messages"][-1]["content"])
    print("\n" + "="*50 + "\n")
    
    # Example 2: Question answerable without search
    print("💡 Example 2: General knowledge question")
    inputs = {"messages": [{"role": "user", "content": "Explain how photosynthesis works"}], "needs_search": False}
    result = app.invoke(inputs)
    print("Answer:", result["messages"][-1]["content"])
