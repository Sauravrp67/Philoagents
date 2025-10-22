from functools import lru_cache

from langgraph.graph import END,START,StateGraph
from langgraph.prebuilt import tools_condition

from edges import (
    should_summarize_conversation
                   )
from state import PhilosopherState
from nodes import (
    conversation_node,
    summarize_context_node,
    summarize_conversation_node,
    connector_node
    )

@lru_cache(maxsize=1)
def create_workflow_graph():
    graph_builder = StateGraph(PhilosopherState)

    #Add all nodes
    graph_builder.add_node("conversation_node",conversation_node)
    graph_builder.add_node("summarize_conversation_node",summarize_conversation_node)
    graph_builder.add_node("summarize_context_node",summarize_context_node)
    graph_builder.add_node("connector_node",connector_node)
    
    #Define the flow
    graph_builder.add_edge(START,"conversation_node")
    graph_builder.add_conditional_edges(
        "conversatoin_node",
        tools_condition
        {
            "tools": ""
        }
        
        )

    return graph_builder

