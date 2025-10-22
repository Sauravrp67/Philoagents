from state import PhilosopherState
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode

def conversation_node(state:PhilosopherState,config:RunnableConfig):
    pass

def connector_node(state:PhilosopherState):
    pass

def summarize_conversation_node(state:PhilosopherState,config:RunnableConfig):
    pass

def summarize_context_node(state:PhilosopherState,config:RunnableConfig):
    pass