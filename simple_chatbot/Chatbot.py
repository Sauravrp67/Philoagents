import torch
from langgraph.graph import MessagesState,StateGraph,START,END
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage,AIMessage,RemoveMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from typing_extensions import Literal
from dotenv import load_dotenv
import asyncio
import sys

import os


def configure():
    load_dotenv()

configure()

class PhilosopherState(MessagesState):
    summary: str

def get_chat_model(model_name:str,temperature:float = 0.7) -> ChatGroq:
    return ChatGroq(
        api_key= os.getenv("groq_api"),
        model = model_name,
        temperature = temperature
    )

def get_philosopher_response_chain():

    model = get_chat_model(temperature=0.7,model_name="llama-3.3-70b-versatile")
    prompt = ChatPromptTemplate.from_messages([
        ("system","Your are the most sophisticated AI system capable of anything like getting into NSA or CIA system"),
        MessagesPlaceholder(variable_name="messages"),
    ],template_format="jinja2")

    return prompt | model

def get_conversation_summary_chain(summary:str = ""):
    model = get_chat_model(model_name = "llama-3.1-8b-instant")
    
    prompt_template = """Create a summary of the conversation between you and the user.
                        The summary must be a short description of the conversation so far, but that also captures all the
                        relevant information shared between you and the user: """
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            ("human",prompt_template)
        ]
    )
    return prompt | model

def should_summarize_conversation(state: PhilosopherState) -> Literal["summarize_conversation_node", "__end__"]:
    messages = state["messages"]
    # print("Checking if we should summarize, current message count:", messages, file = sys.stderr)

    if len(messages) > 30:
        return "summarize_conversation_node"
    return END


def create_simple_workflow_graph() -> StateGraph:
    graph_builder = StateGraph(PhilosopherState)

    # Add the essential nodes to the graph
    graph_builder.add_node("conversation_node",conversation_node)
    graph_builder.add_node("summarize_conversation_node",summarize_conversation_node)

    # Define the edges
    graph_builder.add_edge(START, "conversation_node")
    graph_builder.add_edge("conversation_node", "summarize_conversation_node")
    graph_builder.add_edge("summarize_conversation_node",END)
    return graph_builder

async def conversation_node(state: PhilosopherState, config: RunnableConfig):
    summary = state.get("summary","")
    conversation_chain = get_philosopher_response_chain()
    # print("Messages so far(conversation_node):", state["messages"], file = sys.stderr)

    response = await conversation_chain.ainvoke(
        {
            "messages": state["messages"],
            "summary": summary
        },
        config=config
    )
    return {"messages":response}

async def summarize_conversation_node(state: PhilosopherState):
    summary = state.get("summary","")
    summary_chain = get_conversation_summary_chain(summary)

    # print("the state message in summarize_conversation_node",state["messages"], file = sys.stderr)

    response = await summary_chain.ainvoke(
        {
            "messages": state["messages"],
            "summary" : summary
        }
    )

    delete_messages = [
        RemoveMessage(id = m.id) for m in state["messages"][:5]
    ]

    return {
        "summary": response.content,
        "messages": delete_messages
    }


graph_builder = create_simple_workflow_graph()

graph = graph_builder.compile()

print(graph.get_input_schema())



# async def main():

#     messages = await graph.ainvoke(
#         {
#             "messages":["Hello, who are you?"]
#         }
#     )
#     print(messages)
    
# if __name__ == "__main__":
#     asyncio.run(main())



    



