from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

def pick_llm(level : str):
    """
    This function picks the appropriate LLM based on the level of the query.
    
    Args:
        level (str): The level of the query. It can be 'simple', 'intermediate', or 'complex'.
        
    Returns:
        ChatAnthropic: The LLM instance to be used for the given level.
    """
    if level.lower() == 'simple':
        llm = ChatAnthropic(model="claude-haiku-5-5")
    elif level.lower() == 'intermediate':
        llm = ChatAnthropic(model="claude-sonnet-5-5")
    elif level.lower() == 'complex':
        llm = ChatAnthropic(model="claude-opus-5-5")
    else:
        raise ValueError("Invalid level. Please choose from 'simple', 'intermediate', or 'complex'.")

    return llm

# llm_obj = pick_llm("simple")  # Example usage, you can change the level as needed
# print(llm_obj.invoke("What is the capital of France?"))  # Example invocation, you can change the prompt as needed