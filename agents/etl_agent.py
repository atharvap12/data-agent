import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from utils.etl_tools import ETLTools
from models.schema import ETLAgentSchema
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from utils.logger import get_logger, log_node, block

logger = get_logger("etl_agent")


#------------------------------------ AGENT TOOLS ------------------------------------#


@tool
def extract_load_tool(url:str, output_folder:str, format:str) -> str:
    """
    This tool extracts the data from the API (url) and loads it into the
    the desired location (output_folder).

    Args:
        url (str): The API endpoint from which to extract data.
        output_folder (str): The folder where the extracted data will be saved.
        format (str): The format in which to save the extracted data (csv, json, parquet).
    
    Returns:
        str: A message indicating the success or failure of the operation.

    """
    logger.info(f"[extract_load_tool] Extracting from {url} → folder '{output_folder}' as {format}")
    etl_tools = ETLTools()
    result = etl_tools.extract_load(url, output_folder, format)
    logger.info(f"[extract_load_tool] Result: {result}")
    return result


@tool
def transform_load_tool(input_file_path:str,output_folder:str,output_format:str, user_question:str) -> str:
    """
    This tool transforms the data from the specified file and loads it into the
    desired location (output_folder).

    Args:
        input_file_path (str): The path to the file containing the data to be transformed.
        output_folder (str): The folder where the transformed data will be saved.
        output_format (str): The format in which to save the transformed data (csv, json, parquet).
    
    Returns:
        str: A message indicating the success or failure of the operation.

    """
    logger.info(f"[transform_load_tool] Transforming '{input_file_path}' → folder '{output_folder}' as {output_format}")
    logger.info(f"[transform_load_tool] Transformation asked for: {user_question}")

    etl_tools = ETLTools()

    top_3_rows = etl_tools.transform_load_context(input_file_path)

    llm = pick_llm("intermediate")

    prompt = f"""
            You are a Python Data Analyst who uses Pandas to analyze data. 
            You need to provide only the Pandas Code that will help to perform the right ETL operations on the data stored in the file : {input_file_path}
            as per the user's question. Do not provide any explanation or comments, only
            the code should be provided. The code should be in a format that can be executed 
            in a Python environment with Pandas installed. 
            Don't write anything else than Pandas Code. \n
            
            Create the Pandas Dataframe from the data stored in the file : {input_file_path} and then 
            write the code to transform and save the data at {output_folder}.
            Here's the user's question: {user_question}\n
            Here's the context of the data you will be analyzing: {top_3_rows}\n

        """

    response = llm.invoke(prompt).text

    # Cleaning: the LLM often wraps its code in markdown fences like
    #     ```python
    #     import pandas as pd
    #     ...
    #     ```
    # and those fences must be removed before we can run the code.
    #
    # BEFORE: we used `.strip('```').strip().lstrip('python')`. The catch is that
    # `lstrip('python')` does NOT remove the word "python". It removes any of the
    # letters p, y, t, h, o, n from the start, one by one, until it finds a
    # different letter. So if the code started with `pd.read_csv(...)`, the "p"
    # was removed and we ran `d.read_csv(...)`, which is broken code.
    #
    # NOW: the regex below removes only a real opening fence (``` or ```python)
    # at the very start and a closing ``` at the very end. Everything else in the
    # code is left exactly as the LLM wrote it.
    pandas_code = re.sub(r"^```(?:python)?\s*|\s*```$", "", response.strip())

    logger.info(f"[transform_load_tool] Generated pandas code:{block(pandas_code)}")

    # Execute the Pandas code
    results = etl_tools.execute_code(pandas_code)

    if results.startswith("Failed"):
        logger.error(f"[transform_load_tool] Code execution: {results}")
    else:
        logger.info(f"[transform_load_tool] Code execution: {results}")

    return f"The data is transformed and saved at {output_folder} in {output_format} format. \n\n Pandas Code Executed: \n {pandas_code} \n\n Execution Result: \n {results}"


# Toolkit 
tools = [extract_load_tool, transform_load_tool]

llm = pick_llm("intermediate")
llm_bind = llm.bind_tools(tools)


# ---------------------------------------- AGENT GRAPH ---------------------------------------- #

@log_node(logger)
def llm_node(state:ETLAgentSchema):

    messages = state.messages

    prompt = f"""
            You are a Python Data Analyst who has access to tools that can extract and load, 
            transform and load data. You will be provided with a user's question 
            and you would need to perform the right ETL operations as per the user's question. 
            If the operation is performed then inform the user and end the coversation.
            Here's the chat history: {messages}\n
    """

    final_answer = llm_bind.invoke(prompt)

    # Log what the LLM decided: call tool(s), or answer the user and finish
    if final_answer.tool_calls:
        for tool_call in final_answer.tool_calls:
            logger.info(f"[llm_node] LLM decided to call tool '{tool_call['name']}' with args {tool_call['args']}")
    else:
        logger.info(f"[llm_node] LLM answered without calling a tool:{block(final_answer.text)}")

    state.messages = messages + [final_answer]

    return state


@log_node(logger)
def tool_node(state:ETLAgentSchema):
    """
    This node is responsible for invoking the appropriate tool based on the user's question and the context provided by the LLM.
    """

    tools_results = []

    tools_by_name = {tool.name: tool for tool in tools}

    tool_calls = state.messages[-1].tool_calls

    for tool_call in tool_calls:

        tool = tools_by_name[tool_call['name']]
        logger.info(f"[tool_node] Running tool '{tool_call['name']}'")
        observation = tool.invoke(tool_call['args'])

        tools_results.append(ToolMessage(content=observation, tool_call_id = tool_call['id']))

    state.messages = state.messages + tools_results

    return state   


# Nodes & Edges
etl_analyst_graph = StateGraph(ETLAgentSchema)
etl_analyst_graph.add_node("llm_node", llm_node)
etl_analyst_graph.add_node("tool_node", tool_node)

etl_analyst_graph.add_edge(START, "llm_node")

def is_tool_call(state:ETLAgentSchema):
    tool_calls = state.messages[-1].tool_calls

    if tool_calls:
        logger.info(f"[is_tool_call] Decision: {len(tool_calls)} tool call(s) requested → go to 'tool_node'")
        return "tool_node"
    else:
        logger.info("[is_tool_call] Decision: no tool calls → END (ETL agent is done)")
        return "end"

etl_analyst_graph.add_conditional_edges(
    "llm_node",is_tool_call,
    {
        "tool_node": "tool_node",
        "end": END
    }
)

etl_analyst_graph.add_edge("tool_node", "llm_node")

etl_analyst = etl_analyst_graph.compile()

if __name__ == "__main__":
    # Compile the Graph
    

    # Optional
    from IPython.display import display, Image
    img = Image(etl_analyst.get_graph().draw_mermaid_png())
    with open("etl_analyst_graph.png", "wb") as f:
        f.write(img.data)

    response = etl_analyst.invoke(
        {"messages":[HumanMessage(content="I want to extract the data from the API endpoint 'https://pokeapi.co/api/v2/pokemon' and save it to data/extract folder in the csv folder")]}
    )

#     response = etl_analyst.invoke(
#          {"messages":[HumanMessage(content=f"""
#             I want to transform the data stored in the 'c:\\Data_Agent\\data\\extract\\extracted_data.csv' file 
#             and save the transformed data in the 'c:\\Data_Agent\\data\\transform' folder in the csv format.
#             The transformation should filter the data to show bulbasaur pokemon only.
# """)]}
#     )    

    print(response)