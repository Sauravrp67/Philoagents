from langgraph.graph import MessagesState

#What does it mean when we say injecting a field dynamically vs statically?
#Dynamically means that the field is added to the state during the execution of the workflow
#Statically means that the field is added to the state when we define the workflow graph
class PhilosopherState(MessagesState):
    philosopher_context: str #dynamically injected by RAG tool
    philosopher_name: str #statically injected when we define the graph
    philosopher_perspective: str #statically injected when we define the graph
    philosopher_style: str #statically injected when we define the graph
    summary: str #dynamically injected by summarize_conversation_node

def state_to_str(state:PhilosopherState) -> str:
    if "summary" in state and bool(state["summary"]):
        conversation = state["summary"]
    elif "message" in state and bool(state["messages"]):
        conversation = state["messages"]
    else:
        conversation = ""
    return f"""
    PhilosopherState(philosopher_context={state["philosopher_context"]}, 
    philosopher_name={state["philosopher_name"]}, 
    philosopher_perspective={state["philosopher_perspective"]}, 
    philosopher_style={state["philosopher_style"]}, 
    conversation={conversation})    
"""

if __name__ == "__main__":
    # Example usage
    #Dynamically injecting philosopher_context
    state = PhilosopherState(philosopher_name="Socrates", philosopher_perspective="Ethics", philosopher_style="Socratic")
    state["philosopher_context"] = "The unexamined life is not worth living."
    state["summary"] = "A conversation about the importance of self-examination."
    state_str = state_to_str(state)
    print(state_str)