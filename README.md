# AI Agents with LangChain & LangGraph

A comprehensive project demonstrating intelligent AI agents built with LangChain and LangGraph, featuring Claude AI, web search capabilities, and real-time weather information.

## 🎯 Project Overview

This project showcases advanced agent architectures including:
- **ReAct Agents** - LangChain agents that reason and act iteratively
- **LangGraph Agents** - Graph-based agents with state management
- **Multi-Tool Integration** - Web search and weather tools
- **Interactive Chat** - Real-time conversation with streaming responses
- **Weather Intelligence** - Real-time weather using Open-Meteo API
- **Web Search** - Current information via Tavily Search API

## ✨ Features

✅ **Claude AI Integration** - Uses Anthropic's Claude Opus for intelligent reasoning  
✅ **Web Search** - Real-time search for current events and information  
✅ **Weather Tool** - Get weather for any location with detailed reports  
✅ **Interactive Chat** - Talk to the agent naturally with streaming responses  
✅ **ReAct Pattern** - Reason-then-Act approach for complex tasks  
✅ **Graph Visualization** - See the agent's decision flow and architecture  
✅ **Multiple Agent Types** - Choose from different agent implementations  

## 📦 Installation

### Prerequisites
- Python 3.13+
- Virtual environment (venv or conda)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/bpeddi/aiagents-langchain.git
   cd aiagents-langchain
   ```

2. **Create and activate virtual environment**
   ```bash
   # Using uv (recommended)
   uv venv .venv --python 3.13
   .\.venv\Scripts\Activate.ps1

   # Or using venv
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   # Copy template and edit with your API keys
   cp .env.template .env
   ```
   
   Add your API keys to `.env`:
   ```
   ANTHROPIC_API_KEY=your_anthropic_key_here
   TAVILY_API_KEY=your_tavily_key_here
   OPENAI_API_KEY=your_openai_key_here  # Optional, for GPT models
   ```

## 🚀 Quick Start

### Interactive Agent Chat

Run the main ReAct agent with streaming responses:

```bash
python lanchain_agent.py
```

Then ask questions interactively:
```
👤 You: Where is Super Bowl 2026 being played and what's the weather there?
```

### Standalone Scripts

**LangGraph Agent** - State-based agent with conditional routing:
```bash
python langgraph_agent.py
```

**Weather Tool** - Get weather for any location:
```bash
python get_weather.py
```

**Main Demo** - Run all examples:
```bash
python main.py
```

## 📁 Project Structure

```
.
├── lanchain_agent.py         # Main ReAct agent (interactive)
├── langgraph_agent.py        # LangGraph state-based agent
├── get_weather.py            # Weather tool demonstration
├── main.py                   # Multi-example runner
├── tools.py                  # Tool definitions (weather, search)
├── myagent_notebook.ipynb    # Jupyter notebook version
├── requirements.txt          # Python dependencies
├── .env.template             # Environment variables template
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## 🛠️ Tools Available

### 1. **Web Search Tool** (`search_tool`)
Retrieves current information from the web using Tavily API.

**Use for:**
- News and current events
- Stock prices and market data
- Sports scores
- Product releases
- Real-time information

### 2. **Weather Tool** (`get_weather`)
Fetches detailed weather information using Open-Meteo API.

**Returns:**
- Current temperature and conditions
- Humidity and wind speed
- Precipitation levels
- Day/night indicator

## 📚 Usage Examples

### Example 1: Current Events Question
```python
from lanchain_agent import agent_executor
from langchain_core.messages import HumanMessage

query = "What were the major AI announcements at CES 2026?"
agent_executor.stream({"messages": [HumanMessage(content=query)]})
```

### Example 2: Multi-Tool Query
```python
query = "Where is Super Bowl 2026 and how's the weather there?"
# Agent automatically:
# 1. Searches for Super Bowl location
# 2. Calls weather tool for that location
# 3. Provides comprehensive answer
```

### Example 3: Weather Only
```python
from tools import get_weather

weather = get_weather("San Francisco")
print(weather)
```

## 🔄 Agent Workflow

### ReAct Agent Flow
```
User Query → LLM Decision → Tool Selection → Tool Execution → Analysis → Response
```

### LangGraph Agent Flow
```
User Query → Decide (Need Search?) 
          ↓
         YES → Search → Respond
         NO  → Respond Directly
```

## 🧠 How Agents Work

1. **Understand** - Agent reads your query
2. **Reason** - Decides which tools are needed
3. **Act** - Calls appropriate tools (search, weather, etc.)
4. **Reflect** - Analyzes tool results
5. **Answer** - Provides comprehensive response

## 🔑 API Keys Required

### Anthropic API
- Sign up at https://console.anthropic.com
- Get your API key from the dashboard

### Tavily API (Web Search)
- Sign up at https://tavily.com
- Get your API key from account settings
- Free tier includes generous search limits

### OpenAI API (Optional)
- Only needed if using GPT models instead of Claude
- Sign up at https://platform.openai.com

## 📊 Visualization

View the agent's decision graph:
```python
python lanchain_agent.py
```

The agent will display its architecture:
```
┌─────────────┐
│  User Input │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Agent Logic │
└─────┬───┬───┘
      │   │
      ▼   ▼
   Tool Calls
```

## 🐛 Troubleshooting

### "API Key not found" Error
- Ensure `.env` file exists in project root
- Verify API keys are correctly set
- Check you're using correct key names (ANTHROPIC_API_KEY, TAVILY_API_KEY)

### "Geocoding failed" Error
- Location name not found in database
- Try with major city/state names
- Agent will auto-retry with simplified name

### "TavilySearchResults deprecated" Warning
- This is just a deprecation notice
- Install latest: `pip install -U langchain-tavily`
- Functionality still works

## 📖 Documentation

- **LangChain Docs**: https://python.langchain.com
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **Anthropic Docs**: https://docs.anthropic.com

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📝 License

This project is open source - use freely for learning and development.

## 💡 Advanced Topics

### Custom Tools
Add your own tools to `tools.py`:
```python
@tool
def custom_tool(input: str) -> str:
    """Custom tool description"""
    return "result"
```

### Agent Configuration
Modify LLM settings in agent files:
```python
llm = ChatAnthropic(
    model="claude-opus-4-6",
    temperature=0,  # 0=deterministic, 1=creative
    max_tokens=4096
)
```

### Streaming Mode
Control how responses stream:
```python
agent_executor.stream(
    {"messages": [HumanMessage(content=query)]},
    stream_mode="values"  # or "updates"
)
```

## 🎓 Learning Resources

1. Start with `lanchain_agent.py` for basic usage
2. Explore `tools.py` to understand tool integration
3. Check `myagent_notebook.ipynb` for step-by-step breakdown
4. Run interactive mode: `python lanchain_agent.py`

---

**Built with ❤️ using LangChain, LangGraph, and Claude AI**
