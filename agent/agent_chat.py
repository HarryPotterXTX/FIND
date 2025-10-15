import os
from typing import Annotated
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langchain_tavily import TavilySearch
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.chat_models import ChatTongyi

# APIs
# os.environ["TAVILY_API_KEY"] = "tvly-***"
# os.environ["LLM_API_KEY"] = "sk-***"

# LLM
llm = ChatTongyi(
    model="qwen-max",
    api_key=os.getenv("LLM_API_KEY")
)
tool1 = TavilySearch(max_results=2)
tools = [tool1]
llm_with_tools = llm.bind_tools(tools)

# State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Graph
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert(len(message.tool_calls) <= 1)
    return {"messages": [message]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Run
def stream_graph_updates(user_input: str):
    config = {"configurable": {"thread_id": "1"}}
    events = graph.stream({"messages": [{"role": "user", "content": user_input}]}, config, stream_mode="values")
    for event in events:
        event["messages"][-1].pretty_print()

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        break