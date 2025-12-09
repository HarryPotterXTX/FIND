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
    model="qwen-max-0919",
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

def task_initial():
    while True:
        task = input("Please select the task type (1-Function Discovery, 2-Dimensionless Number Discovery, 3-PDE Discovery): ")
        if task in ['1', '2', '3']:
            break
        else:
            print("Invalid input. Please enter 1, 2, or 3.")
        if task.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            quit()
    if task == '1':
        workflow = """The standard workflow consists of the following steps: 
1. Understand the meaning of each hyperparameter in 'opt/config.yaml'; 
2. Enter the dataset path and the variable to be predicted; 
3. Identify and select the most critical variables as inputs with SHAP; 
4. Utilize the FIND module to discover formulas from the data; 
5. If the formulas are overly complex, invoke the SR module to simplify them; 
6. Validate the derived formulas; 
7. Analyze potential causes of failures; 
8. Generate an experimental report."""
    elif task == '2':
        workflow = """The current task is to discover dimensionless numbers from data.
The standard workflow consists of the following steps: 
1. Understand the meaning of each hyperparameter in 'opt/config.yaml';
2. Enter the dataset path and the variable to be predicted; 
3. Identify and select the most critical variables as inputs with SHAP; 
4. Set Dataset.d=0 to ensure that all discovered latent variables are dimensionless numbers.
5. Utilize the FIND module to discover formulas from the data; 
6. If the formulas are overly complex, invoke the SR module to simplify them; 
7. Validate the derived formulas; 
8. Analyze potential causes of failures; 
9. Generate an experimental report."""
    elif task == '3':
        path = input("Please enter the path to your series data: ")
        workflow = f"""The current task is to discover a unified PDE.
The standard workflow consists of the following steps: 
1. Understand the meaning of each hyperparameter in 'opt/config.yaml';
2. Use SINDy to discover PDEs from '{path}';
3. Select a PDE coefficient to be the output variable and set Dataset.data_path to 'dataset/pde.csv'; 
4. Apply SHAP to identify the most critical variables as model inputs, with the exclusion of a0 to a8; 
5. Utilize the FIND module to discover formulas from the data; 
6. If the formulas are overly complex, invoke the SR module to simplify them; 
7. Repeat steps 3 to 6 to discover the relationship between each PDE coefficient and the system parameters;
8. Summarize the form of the PDE and proceed to validate the derived equation; 
9. Analyze potential causes of failures; 
10. Generate an experimental report."""
        
    prompt = f"""You are an AI scientist who can extract formulas from data. 
I will ask some questions or requests, please provide a brief answer. 
After each time you call a tool, you need to pause and suggest to the 
user what to do next. {workflow}"""
    return prompt

prompt = task_initial()
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