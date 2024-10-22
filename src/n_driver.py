import os
import getpass
import google.generativeai as genai
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool, initialize_agent
from langchain.agents import AgentType
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_number(response):
    """
    This function parses the response to extract the integer number of homicides.
    """
    match = re.search(r'\b\d+\b', response)
    if match:
        return int(match.group(0))
    return None

def modify_response(agent, state, year):
    """
    Function to query the agent for homicide data for a given state and year.
    Retries on failure.
    """
    query = f"What is the number of homicides for year {year} in {state}?"
    try:
        response = agent.run(query)
        print(f"Response for {state} in {year}: {response}")
        return get_number(response)
    except Exception as e:
        print(f"Error querying {state} in {year}: {e}")
        return None

if __name__ == "__main__":
    
    # Environment configuration
    
    os.environ["LANGCHAIN_API_KEY"]=input("Enter passphrase for Langchain:")
    os.environ["LANGCHAIN_TRACING_V2"]="True"
    os.environ["LANGCHAIN_ENDPOINT"]=input("Enter endpoint for Langchain:")
    os.environ["LANGCHAIN_PROJECT"]=input("Enter projectname for Langchain:")
    os.environ["TAVILY_API_KEY"]= input("Enter passphrase for Tavily:")
    os.environ["GOOGLE_CLOUD_API_KEY"]=input("Enter passphrase for Google API:")
    os.environ["GOOGLE_API_KEY"]=input("Enter passphrase for Google API:")
    os.environ["OPENAI_API_KEY"]=input("Enter passphrase for OpenAPI:")
    genai.configure(api_key=os.environ["GOOGLE_CLOUD_API_KEY"])
    
    
    genai.configure(api_key=os.environ["GOOGLE_CLOUD_API_KEY"])
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    #llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    #llm = ChatOpenAI(temperature=0.3, model="gpt-4-1106-preview")
    

    query = input("Ask your question: ")
    # Initialize search tool and agent
    formatted_prompt = f"""
    You are a knowledgeable assistant. Please provide a comprehensive and accurate answer to the following question:
    Question: {query}
    """
    search = TavilySearchResults(max_results=95)
    tools = [search]
    agent = initialize_agent(tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
    
    response = agent.invoke(query)#(formatted_prompt)



    #response = agent.run("What is batched inference means in machine learning ")


    print(response["output"])
    

    # Define states and years
    #states = ["New York", "New Orleans", "Los Angeles"]
    #years = ["2023", "2022", "2021", "2020", "2019"]

    # Dictionary to store homicide statistics
    #stats = {state: {} for state in states}

    """
    # Use ThreadPoolExecutor to parallelize the queries
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_state_year = {
            executor.submit(modify_response, agent, state, year): (state, year)
            for state in states
            for year in years
        }
        
        for future in as_completed(future_to_state_year):
            state, year = future_to_state_year[future]
            try:
                homicide_number = future.result()
                stats[state][year] = homicide_number
            except Exception as exc:
                print(f"Error fetching data for {state} in {year}: {exc}")
    
    # Print the final stats
    print(stats)
    """