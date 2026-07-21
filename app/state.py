from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    This is the object that flows through every node in the graph.

    `messages` holds the full conversation (HumanMessage, AIMessage, ToolMessage...).
    The `add_messages` annotation tells LangGraph: "don't overwrite this list when a
    node returns a new message — append it instead." That's how conversation memory
    works here: each node just returns the *new* message(s), and LangGraph merges
    them into the running history for us.
    """
    messages: Annotated[list, add_messages]
