import os
from dotenv import load_dotenv

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage

st.set_page_config(page_title="Research Buddy", page_icon="📃")

with st.sidebar:
    OPENAI_API_KEY = st.text_input("Your own OPENAI API Key", type="password")
    SERP_API_KEY = st.text_input("Your own SERP API Key", type="password")
    os.environ["SERP_API_KEY"] = SERP_API_KEY
    BROWSERLESS_API_KEY = st.text_input("Your own BROWSERLESS API Key", type="password")
    os.environ["BROWSERLESS_API_KEY"] = BROWSERLESS_API_KEY

st.header("ResearchBuddy 📃")

if OPENAI_API_KEY:
    load_dotenv()
    browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
    serper_api_key = os.getenv("SERP_API_KEY")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 1. Tool for search

def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({"q": query})

    headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text


# search("What is meta's tread product?") # test


# 2. Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "application/json",
    }

    # Define the data to be sent in the request
    data = {"url": url}

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:  # Successfully extreacted content
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENT:", text)

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


# scrape_website("What is langChain", "https://www.langchain.com/") # Test


def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"]
    )

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True,
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


# Handling Multiple inputs from the scrapped website


class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""

    objective: str = Field(
        description="The objective & task that users give to the agent"
    )
    url: str = Field(description="The url of the website to be scraped")


# Handling Multiple url from the scrapped website


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")


# 3. Creating the agent

tools = [
    Tool(
        name="Search",
        func=search,
        description="To answer questions on data and current events. Ask targeted questions",
    ),
    ScrapeWebsiteTool(),
]

# Base Prompt

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ Be as descriptive and structured as possible
            7/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            8/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research, Create a seperate section called 'References' where you will  reference data & links to back up your research."""
)

# Args for agents (can be multiple)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

if OPENAI_API_KEY:
    # Initiliazing an LLM
    llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")

    # Saving the chat in memory
    memory = ConversationSummaryBufferMemory(
        memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000
    )

    # Initializing the main agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,  # Test more
        verbose=True,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )

# frontend streamlit code


def main():
        query = st.text_input("Research goal")
        
        if OPENAI_API_KEY and SERP_API_KEY and BROWSERLESS_API_KEY:
            if st.button("Start research"):
                if query:
                    st.write("Doing research for ", query)

                    result = agent({"input": query})

                    response = st.info(result["output"])

                else:
                    st.warning("Enter a query", icon="⁉️")


if __name__ == "__main__":
    main()

# Test with
# Most notable research papers in the field of Mars Rovers