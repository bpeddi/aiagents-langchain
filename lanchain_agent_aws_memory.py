# pip install langgraph langchain-anthropic tavily-python python-dotenv

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from tools import TOOLS
from langgraph_checkpoint_aws import AgentCoreMemorySaver , AgentCoreMemoryStore
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langgraph.store.base import BaseStore
import uuid


app = BedrockAgentCoreApp()
# Load environment variables
load_dotenv()

# Validate API keys
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
MEMORY_ID = "memory_for_lagchain_agent-ZgBqYyGADI"

if not anthropic_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables. Please set it in your .env file.")
if not tavily_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables. Please set it in your .env file.")

print("✅ API keys loaded successfully!")

# Initialize checkpointer for state persistence. No additional setup required.
# Sessions will be saved and persisted for actor_id/session_id combinations
checkpointer = AgentCoreMemorySaver(memory_id=MEMORY_ID)
# Initialize store for saving and searching over long term memories
# such as preferences and facts across sessions
store = AgentCoreMemoryStore(memory_id=MEMORY_ID)
# Initialize Tavily search tool with custom description for better tool selection

class MemoryMiddleware(AgentMiddleware):
    # Pre-model hook: saves messages and retrieves long-term memories
    def pre_model_hook(self, state: AgentState, config: RunnableConfig, *, store: BaseStore):
        """
        Hook that runs before LLM invocation to:
        1. Save the latest human message to long-term memory
        2. Retrieve relevant user preferences and memories
        3. Append memories to the context
        """
        actor_id = config["configurable"]["actor_id"]
        thread_id = config["configurable"]["thread_id"]
        
        # Namespace for this specific session
        namespace = (actor_id, thread_id)
        messages = state.get("messages", [])
        
        # Save the last human message to long-term memory
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                store.put(namespace, str(uuid.uuid4()), {"message": msg})
                
                # OPTIONAL: Retrieve user preferences from long-term memory
                # Search across all sessions for this actor
                user_preferences_namespace = ("preferences", actor_id)
                try:
                    preferences = store.search(
                        user_preferences_namespace, 
                        query=msg.content, 
                        limit=5
                    )
                    
                    # If we found relevant memories, add them to the context
                    if preferences:
                        memory_context = "\n".join([
                            f"Memory: {item.value.get('message', '')}" 
                            for item in preferences
                        ])
                        # You can append this to the messages or use it another way
                        print(f"Retrieved memories: {memory_context}")
                except Exception as e:
                    print(f"Memory retrieval error: {e}")
                break
        
        return {"messages": messages}


    # OPTIONAL: Post-model hook to save AI responses
    def post_model_hook(state, config: RunnableConfig, *, store: BaseStore):
        """
        Hook that runs after LLM invocation to save AI messages to long-term memory
        """
        actor_id = config["configurable"]["actor_id"]
        thread_id = config["configurable"]["thread_id"]
        namespace = (actor_id, thread_id)
        
        messages = state.get("messages", [])
        
        # Save the last AI message
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                store.put(namespace, str(uuid.uuid4()), {"message": msg})
                break
        
        return state



# ============ 1. Initialize LLM and Tools ============
llm = ChatAnthropic(
    model="claude-opus-4-6",  # Best for agents (Haiku is faster/cheaper alternative)
    temperature=0,
    max_tokens=4096,
    timeout=None,
    max_retries=2,
)



tools=[*TOOLS]

system_prompt = """You are a helpful FAQ assistant with access to a knowledge base and user memory.

Your goal is to answer user questions accurately using the available tools while remembering user preferences.

Guidelines:
1. Check if you have relevant user preferences or history from previous conversations
2. Use the tools to find relevant information from the knowledge base
3. If the query is complex, use reformulate_query to search different aspects
4. Personalize responses based on user preferences when relevant
5. Always provide a clear, concise answer based on the retrieved information
6. If you cannot find relevant information, clearly state that

Think step-by-step and use tools strategically to provide the best answer."""

# ============ 2. Create ReAct Agent ============
# This creates a standard ReAct agent that:
# - Uses LLM to decide when to call tools
# - Handles tool execution and response formatting automatically
# - Maintains conversation history in MessagesState format
agent_executor = create_agent(
    model=llm, 
    tools=tools,
    checkpointer=checkpointer ,
    store=store,
    middleware=[MemoryMiddleware()],
    system_prompt=system_prompt  

    )

# # ============ 3. Visualize the Graph ============
# print("\n📊 Agent Graph Structure (ReAct Pattern):")
# print("=" * 60)
# try:
#     # ASCII visualization (works without dependencies)
#     print(agent_executor.get_graph().draw_ascii())
# except Exception as e:
#     print(f"⚠️  ASCII visualization failed: {e}")
#     print("💡 Tip: Install graphviz for PNG/SVG exports: `pip install graphviz`")

# ============ 4. Helper: Print message chunks clearly ============
def print_stream_chunk(chunk, step_num):
    """Print different message types during streaming"""
    if "messages" not in chunk:
        return
    
    new_messages = chunk["messages"]
    if not new_messages:
        return
    
    msg = new_messages[-1]
    
    # Tool call decision (LLM wants to use a tool)
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        print(f"\n[Step {step_num}] 🤖 Agent decided to call tool:")
        for tool_call in msg.tool_calls:
            print(f"   🔧 {tool_call['name']}({tool_call['args']})")
    
    # Tool response (results from Tavily)
    elif isinstance(msg, ToolMessage):
        print(f"\n[Step {step_num}] 🔍 Tool response received:")
        # Truncate long responses for readability
        content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
        print(f"   💡 {content}")
    
    # Final assistant response
    elif isinstance(msg, AIMessage) and msg.content and not getattr(msg, 'tool_calls', None):
        print(f"\n[Step {step_num}] ✅ FINAL ANSWER:")
        print(f"\n🤖 {msg.content}")
# # Optional: Save Mermaid diagram for documentation
# try:
#     mermaid = agent_executor.get_graph().draw_mermaid()
#     with open("react_agent_graph.mmd", "w") as f:
#         f.write(mermaid)
#     print("\n✅ Mermaid diagram saved to 'react_agent_graph.mmd'")
# except:
#     pass

# =================== Bedrock Agentcore integration ==================
# 
@app.entrypoint
def agent_invocation(payload, context):
    """
    Handler for agent invocation in AWS Bedrock AgentCore
    
    Args:
        payload: Dict containing 'prompt' key with user query
        context: AWS Lambda context object
    
    Returns:
        Dict with 'result' key containing agent response
    
    Expected payload format:
    {
        "prompt": "Your question here"
    }
    """
    try:
        # Validate payload
        if not payload or not isinstance(payload, dict):
            return {
                "error": "Invalid payload format. Expected JSON dict with 'prompt' key",
                "result": None
            }
        
        # Extract user message
        user_message = payload.get("prompt", "").strip()
            # Extract or generate actor_id and thread_id
        actor_id = payload.get("actor_id", "default-user")
        thread_id = payload.get("thread_id", payload.get("session_id", "default-session"))
    
    # Configure memory context
        config = {
            "configurable": {
                "thread_id": thread_id,  # Maps to AgentCore session_id
                "actor_id": actor_id     # Maps to AgentCore actor_id
            }
        }

        if not user_message:
            return {
                "error": "No prompt found in input. Please provide a 'prompt' key with your question.",
                "result": None
            }
        
        print(f"🔍 Processing user query: {user_message}")
        # print(f"📋 AWS Context: {context.function_name if context else 'No context'}")
        
        # Initialize variables
        step = 0
        final_answer = None
        
        # Stream all intermediate steps
        try:
            for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=user_message)]},config=config,
                stream_mode="values"
            ):
                step += 1
                print_stream_chunk(chunk, step)
                
                # Capture final answer for summary
                if "messages" in chunk:
                    last_msg = chunk["messages"][-1]
                    if isinstance(last_msg, AIMessage) and last_msg.content and not getattr(last_msg, 'tool_calls', None):
                        final_answer = last_msg.content
        
        except Exception as e:
            error_msg = f"Error during agent execution: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "error": error_msg,
                "result": None
            }
        
        # Validate final answer
        if not final_answer:
            return {
                "error": "Agent failed to generate a response",
                "result": None
            }
        
        print(f"\n✅ Agent completed in {step} steps")
        print(f"📤 Final answer: {final_answer[:200]}..." if len(final_answer) > 200 else f"📤 Final answer: {final_answer}")
        
        return {
            "result": final_answer,
            "steps": step,
            "success": True
        }
    
    except Exception as e:
        error_msg = f"Unexpected error in agent_invocation: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "error": error_msg,
            "result": None,
            "success": False
        }


# Run the Bedrock AgentCore app
if __name__ == "__main__":
    app.run() 


# # ============ 4. Run the Agent ============
# if __name__ == "__main__":
#     print("\n" + "="*60)
#     print("🚀 AGENT READY - Claude ReAct Agent with Streaming")
#     print("="*60)
#     print("\n💬 Interactive Agent Chat")
#     print("Type 'exit' or 'quit' to end the conversation\n")
    
#     # Interactive conversation loop
#     while True:
#         try:
#             # Get user input
#             user_query = input("👤 You: ").strip()
            
#             # Check for exit commands
#             if user_query.lower() in ['exit', 'quit', 'bye']:
#                 print("\n👋 Goodbye! Thanks for using the agent.")
#                 break
            
#             # Skip empty inputs
#             if not user_query:
#                 print("⚠️  Please enter a question.\n")
#                 continue
            
#             print("\n" + "-" * 60)
#             step = 0
#             final_answer = None
            
#             # Stream all intermediate steps
#             for chunk in agent_executor.stream(
#                 {"messages": [HumanMessage(content=user_query)]},
#                 stream_mode="values"
#             ):
#                 step += 1
#                 print_stream_chunk(chunk, step)
                
#                 # Capture final answer for summary
#                 if "messages" in chunk:
#                     last_msg = chunk["messages"][-1]
#                     if isinstance(last_msg, AIMessage) and last_msg.content and not getattr(last_msg, 'tool_calls', None):
#                         final_answer = last_msg.content
            
#             print("\n" + "-" * 60 + "\n")
        
#         except KeyboardInterrupt:
#             print("\n\n👋 Agent interrupted. Goodbye!")
#             break
#         except Exception as e:
#             print(f"\n❌ Error: {str(e)}")
#             print("Please try another question.\n")

  