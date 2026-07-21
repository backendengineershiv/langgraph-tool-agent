import os
from pathlib import Path

from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from app.state import AgentState
from app.tools import TOOLS


def _load_env() -> None:
    """Load environment variables from the workspace .env file if present."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()


required_env = [
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
]
missing_env = [name for name in required_env if not os.getenv(name)]
if missing_env:
    raise RuntimeError(
        "Missing required Azure OpenAI environment variables: " + ", ".join(missing_env)
    )

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
    temperature=0,
).bind_tools(TOOLS)


def agent_node(state: AgentState) -> dict:
    """Calls the LLM with the full message history so far."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # gets appended, not overwritten


tool_node = ToolNode(TOOLS)
# ToolNode is a prebuilt LangGraph node: it looks at the last AI message,
# finds any tool_calls on it, executes the matching Python functions from
# TOOLS, and returns their results as ToolMessages. Saves us writing this by hand.


# --- 3. Conditional edge -----------------------------------------------------------
def should_continue(state: AgentState) -> str:
    """
    This is the crux of LangGraph: after the agent responds, we inspect the
    result and decide where to go next. This is a plain Python if/else —
    LangGraph just calls it for us after every 'agent' node run.
    """
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"   # LLM asked for a tool -> go run it
    return END            # LLM answered directly -> we're done


# --- 4. Wire the graph together -----------------------------------------------------------
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point("agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")  # after running a tool, loop back to the agent
# so it can turn the tool's result into a final natural-language answer.

# MemorySaver = an in-memory checkpointer. It saves the state after every node
# run, keyed by a "thread_id". That's what gives us multi-turn memory per
# conversation without us managing any state ourselves.
checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)
