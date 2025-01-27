from typing import Annotated
from langchain_core.tools import tool
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from typing import Literal, List
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import MessagesState, END
from langgraph.types import Command
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers."""
    return a % b

# Pydantic
class Router(BaseModel):
    """Worker to act next given user request."""

    next: str = Field(description="The name of worker to act next")


members = ["multiply", "add", "divide"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {options}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results. When finished,"
    " respond with FINISH."
)

# Load the .env file
load_dotenv()

#Set Ggoogle api key
os.environ["GOOGLE_API_KEY"]=os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")

structured_llm = llm.with_structured_output(Router)

class State(TypedDict):
    next: str
    messages: Annotated[List, add_messages]


def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    # response = llm.with_structured_output(Router).invoke(messages)
    response = structured_llm.invoke(messages)
    goto = response.next
    if goto == "FINISH":
        goto = END


    return Command(goto=goto, update={"next": goto})




multiply_agent = create_react_agent(
    llm, tools=[multiply])


def multiply_node(state: State) -> Command[Literal["supervisor"]]:
    result = multiply_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="multiply")
            ]
        },
        goto="supervisor",
    )


# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
add_agent = create_react_agent(llm, tools=[add])


def add_node(state: State) -> Command[Literal["supervisor"]]:
    result = add_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="add")
            ]
        },
        goto="supervisor",
    )

divide_agent = create_react_agent(llm, tools=[add])


def divide_node(state: State) -> Command[Literal["supervisor"]]:
    result = add_agent.invoke(state)
    return Command(
        update={
            "messages":[
                HumanMessage(content=result["messages"][-1].content, name="divide")
            ]
        },
        goto="supervisor",
    )


builder = StateGraph(State)
builder.add_node("supervisor", supervisor_node)
builder.add_node("multiply", multiply_node)
builder.add_node("add", add_node)
builder.add_node("divide", divide_node)
builder.add_edge(START, "supervisor")
graph = builder.compile()


for s in graph.stream(
    {"messages": [("user", "What's the value of 8 + 2 / 2")]}
):
    print(s)
    print("----")

