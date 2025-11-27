import os, sys
from typing import Annotated
from omegaconf import OmegaConf
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langchain_tavily import TavilySearch
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.chat_models import ChatTongyi
from langgraph.prebuilt import ToolNode, tools_condition

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from agent.tools import tools

# APIs
# os.environ["TAVILY_API_KEY"] = "tvly-***"
# os.environ["LLM_API_KEY"] = "sk-***"

# LLM
llm = ChatTongyi(
    model="qwen-max-latest",
    api_key=os.getenv("LLM_API_KEY")
)

llm_with_tools = llm.bind_tools(tools)

# Configuration
config_path = 'opt/agent_template.yaml'
opt = OmegaConf.load(config_path)
OmegaConf.save(opt, 'opt/agent_instance.yaml', resolve=True)

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

workflow = """The standard workflow consists of the following steps: 
0. Understand the meaning of each hyperparameter in opt/config.yaml; 
1. Enter the dataset path and the variable to be predicted; 
2. Identify and select the most critical variables as inputs with SHAP; 
3. Utilize the FIND module to discover formulas from the data; 
4. If the formulas are overly complex, invoke the SR module to simplify them; 
5. Validate the derived formulas; 
6. Analyze potential causes of failures; 
7. Generate an experimental report."""

prompt = f"""You are an AI scientist who can extract formulas from data. 
I will ask some questions or requests, please provide a brief answer. 
After each time you call a tool, you need to pause and suggest to the 
user what to do next. {workflow}"""

stream_graph_updates(prompt)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        break