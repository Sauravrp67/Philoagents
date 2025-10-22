from state import PhilosopherState
from typing_extensions import Literal
from langgraph.graph import END


def should_summarize_conversation(state:PhilosopherState) -> Literal['summarize_conversation_node', '__end__']:
    messages = state["messages"]

    if len(messages) > 30:
        return "summarize_conversation_node"
    return END

