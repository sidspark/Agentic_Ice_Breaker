import os
from dotenv import load_dotenv
from langchain.chains.summarize.refine_prompts import prompt_template

from utils.tools import get_profile_url_tavily

load_dotenv()

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor
)
from langchain import hub

def lookup(name: str) -> str:
    model_name = "qwen2.5:14b"
    llm = ChatOllama(temperature=0, model=model_name, num_gpu=1)

    template = """
    Given the full name {name_of_person} I want you to get me a link to their LinkenIn profile page.
    Your answer should contain only a URL
    """

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url_tavily,
            description="useful for when you need to get LinkedIn Page URL"
        )
    ]

    react_prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input={
            "input": prompt_template.format_prompt(name_of_person=name)
        }
    )

    url = result["output"]
    return url