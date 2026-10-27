#if I direcntly run this python file, it will consider the /agents folder as the root folder and will not be able to find the utils folder. So, I will add the parent directory to the sys.path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.llm_pick import pick_llm
from models.schema import AgentSchema

# -------------------------------------------------AI Agent for SQL Query Generation-------------------------------------------------

def curate_question(state: AgentSchema) -> AgentSchema:
    user_question = state.user_ques #we can't access it like state['user_question'] because state is a pydantic model and not a dictionary.
    llm = pick_llm("simple") 
    curated_question = llm.invoke(f"Curate the following question for SQL query generation: {user_question}")
    state.curated_ques = curated_question
    return state

def prepare_prompt_query_context(state: AgentSchema) -> AgentSchema:
    curated_question = state.curated_ques
    pass
