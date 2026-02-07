# pip install langgraph langchain-anthropic tavily-python python-dotenv

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from tools import TOOLS

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Load environment variables
load_dotenv()

# Validate API keys
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

if not anthropic_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables. Please set it in your .env file.")
if not tavily_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables. Please set it in your .env file.")

print("✅ API keys loaded successfully!")

# ============ 1. Initialize LLM and Tools ============
llm = ChatAnthropic(
    model="claude-opus-4-6",  # Best for agents (Haiku is faster/cheaper alternative)
    temperature=0,
    max_tokens=4096,
    timeout=None,
    max_retries=2,
)



# Initialize Tavily search tool with custom description for better tool selection


tools=[*TOOLS]

# ============ 2. Create ReAct Agent ============
# This creates a standard ReAct agent that:
# - Uses LLM to decide when to call tools
# - Handles tool execution and response formatting automatically
# - Maintains conversation history in MessagesState format
agent_executor = create_agent(llm, tools)

# ============ 3. Visualize the Graph ============
print("\n📊 Agent Graph Structure (ReAct Pattern):")
print("=" * 60)
try:
    # ASCII visualization (works without dependencies)
    print(agent_executor.get_graph().draw_ascii())
except Exception as e:
    print(f"⚠️  ASCII visualization failed: {e}")
    print("💡 Tip: Install graphviz for PNG/SVG exports: `pip install graphviz`")
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

# ============ 4. Run the Agent ============
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 AGENT READY - Claude ReAct Agent with Streaming")
    print("="*60)
    print("\n💬 Interactive Agent Chat")
    print("Type 'exit' or 'quit' to end the conversation\n")
    
    # Interactive conversation loop
    while True:
        try:
            # Get user input
            user_query = input("👤 You: ").strip()
            
            # Check for exit commands
            if user_query.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Goodbye! Thanks for using the agent.")
                break
            
            # Skip empty inputs
            if not user_query:
                print("⚠️  Please enter a question.\n")
                continue
            
            print("\n" + "-" * 60)
            step = 0
            final_answer = None
            
            # Stream all intermediate steps
            for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=user_query)]},
                stream_mode="values"
            ):
                step += 1
                print_stream_chunk(chunk, step)
                
                # Capture final answer for summary
                if "messages" in chunk:
                    last_msg = chunk["messages"][-1]
                    if isinstance(last_msg, AIMessage) and last_msg.content and not getattr(last_msg, 'tool_calls', None):
                        final_answer = last_msg.content
            
            print("\n" + "-" * 60 + "\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Agent interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try another question.\n")

  