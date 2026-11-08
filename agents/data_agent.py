import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from utils.etl_tools import ETLTools
from models.schema import RouterSchema, DataAgentSchema
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from agents.etl_agent import etl_analyst
from agents.sql_analyst import sql_analyst
from utils.logger import get_logger, log_node

logger = get_logger("data_agent")


llm = pick_llm("simple")

llm_router = llm.with_structured_output(RouterSchema)


# ---------------------------- DATA AGENT GRAPH ---------------------------- #


@log_node(logger)
def router_node(state:DataAgentSchema):

    message = state.messages[-1].content
    logger.info(f"[router_node] User question: {message}")

    prompt = f"""
    You are the router of PipeQL Agent, a data assistant made of two specialist agents.
    Your only job is to read the user's request and decide which agent should handle it.
    You do not answer the request yourself.

    The two agents:

    1. "sql" - SQL Analyst
       Answers questions about the data stored in our PostgreSQL database by writing
       and running a read-only SQL query. The database is for a ride-sharing business
       and has these tables:
       - users: riders and drivers (name, email, phone, city, province, user type,
         signup date, active or not)
       - vehicles: drivers' vehicles (make, model, year, license plate, color, active or not)
       - rides: each trip (rider, driver, request/pickup/dropoff times, pickup and dropoff
         locations, distance in km, fare, surge multiplier, status, cancellation reason)
       - payments: payment for each ride (amount, payment method, payment status,
         transaction id, payment time)
       - ratings: ratings riders give drivers (rating, comment, rated at)
       Choose "sql" when the user wants to look up, count, compare, rank or summarize
       data that lives in the database.
       Examples:
       - "Which 3 users have taken the most rides?"
       - "What payment methods do we support?"
       - "Which city has the most cancelled rides?"
       - "What is the average rating of drivers who drive a Toyota?"

    2. "etl" - ETL Agent
       Builds data pipelines with files and APIs. It can:
       - Extract data from an API endpoint (URL) and save it as a csv, json or parquet file.
       - Transform a data file (csv, json or parquet) with pandas, e.g. filter, clean,
         rename or aggregate it, and save the result to a folder.
       Choose "etl" when the user mentions an API or URL, a file path or file format,
       a folder to save to, or words like extract, load, transform, convert or save.
       Examples:
       - "Extract data from https://pokeapi.co/api/v2/pokemon and save it to data/extract as csv"
       - "Filter data/extract/extracted_data.csv to bulbasaur only and save it in data/transform"

    How to decide:
    - If the request is about files, folders, APIs or moving/reshaping data, choose "etl",
      even if it uses words like "filter" or "show" that sound like a query.
    - If the request is a question about the business data (users, rides, payments,
      ratings, vehicles) and mentions no file or API, choose "sql".
    - If the request is unclear or fits neither agent, choose the closest match and
      say in your comments that the request was ambiguous and why you picked that agent.

    In "comments", explain your decision in one or two sentences.

    User's request:
    {message}
    """

    route_response_dict = llm_router.invoke(prompt).model_dump()

    route_response = route_response_dict['answer']
    logger.info(f"[router_node] Router chose: '{route_response}'")
    logger.info(f"[router_node] Router reason: {route_response_dict['comments']}")

    state.route_response = route_response

    return state

@log_node(logger)
def etl_node(state:DataAgentSchema):

    message = state.messages[-1].content
    logger.info("[etl_node] Handing the question to the ETL agent")

    response = etl_analyst.invoke(
             {"messages":[HumanMessage(content=f"""
            {message}
    """)]}
        ) 
    logger.info("[etl_node] ETL agent finished")
    state.messages = state.messages + [response["messages"][-1]]  # ETL agent's final message

    return state

@log_node(logger)
def sql_node(state:DataAgentSchema):

    message = state.messages[-1].content
    logger.info("[sql_node] Handing the question to the SQL analyst")

    input_schema = {
        "messages": [],
        "user_ques": f"{message}",
        "curated_ques": "",
        "prompt_query_context": "",
        "generated_sql_query": "",
        "is_safe": "No",
        "comments": "",
        "sql_query_execution_result": "",
        "final_answer": ""
    }

    response = sql_analyst.invoke(input_schema)
    logger.info("[sql_node] SQL analyst finished")

    state.messages = state.messages + [AIMessage(content=response["final_answer"])]

    return state




data_agent_graph = StateGraph(DataAgentSchema)

data_agent_graph.add_node("router_node", router_node)
data_agent_graph.add_node("etl_node", etl_node)
data_agent_graph.add_node("sql_node", sql_node)

data_agent_graph.add_edge(START, "router_node")

def route_edge(state: DataAgentSchema) -> str:
    if state.route_response == "sql":
        logger.info("[route_edge] Decision: go to 'sql_node'")
        return "sql_node"
    elif state.route_response == "etl":
        logger.info("[route_edge] Decision: go to 'etl_node'")
        return "etl_node"
    else:
        logger.error(f"[route_edge] Invalid route response: {state.route_response}")
        raise ValueError(f"Invalid route response: {state.route_response}")


data_agent_graph.add_conditional_edges("router_node", route_edge,
                                      {
                                          "sql_node": "sql_node",
                                          "etl_node": "etl_node"
                                      })

data_agent_graph.add_edge("etl_node", END)
data_agent_graph.add_edge("sql_node", END)

data_agent = data_agent_graph.compile()


if __name__ == "__main__":

    # Optional
    from IPython.display import display, Image
    img = Image(data_agent.get_graph().draw_mermaid_png())
    with open("data_agent_graph.png", "wb") as f:
        f.write(img.data)

    response = data_agent.invoke(
        {"messages":[HumanMessage(content="I want to extract the data from the API endpoint 'https://pokeapi.co/api/v2/pokemon' and save it to data/extract folder in the csv folder")],
         "route_response": ""}
    )

    print(response)



