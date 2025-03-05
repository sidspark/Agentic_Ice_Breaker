from dotenv import load_dotenv
from langchain.chains.llm import LLMChain

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup
from output_parser import summary_parser, Summary

from utils.time import calculate_elapsed_time
import time

# information = """
# Jeffrey Preston Bezos (/ˈbeɪzoʊs/ BAY-zohss;[2] né Jorgensen; born January 12, 1964) is an American businessman best known as the founder, executive chairman, and former president and CEO of Amazon, the world's largest e-commerce and cloud computing company. According to Forbes, as of 17 February 2025, Bezos' estimated net worth stood at US$241.9 billion, making him the third richest individual in the world.[3] He was the wealthiest person from 2017 to 2021, according to Forbes and the Bloomberg Billionaires Index.[4][5]
#
# Bezos was born in Albuquerque and raised in Houston and Miami. He graduated from Princeton University in 1986 with degrees in electrical engineering and computer science. He worked on Wall Street in a variety of related fields from 1986 to early 1994. Bezos founded Amazon in mid-1994 on a road trip from New York City to Seattle. The company began as an online bookstore and has since expanded to a variety of other e-commerce products and services, including video and audio streaming, cloud computing, and artificial intelligence. It is the world's largest online sales company, the largest Internet company by revenue, and the largest provider of virtual assistants and cloud infrastructure services through its Amazon Web Services branch.
#
# Bezos founded the aerospace manufacturer and sub-orbital spaceflight services company Blue Origin in 2000. Blue Origin's New Shepard vehicle reached space in 2015 and afterwards successfully landed back on Earth; he flew into space on Blue Origin NS-16 in 2021. He purchased the major American newspaper The Washington Post in 2013 for $250 million and manages many other investments through his venture capital firm, Bezos Expeditions. In September 2021, Bezos co-founded Altos Labs with Mail.ru founder Yuri Milner.[6]
#
# The first centibillionaire on the Forbes Real Time Billionaires Index and the second ever to have achieved the feat since Bill Gates in 1999, Bezos was named the "richest man in modern history" after his net worth increased to $150 billion in July 2018.[7] In August 2020, according to Forbes, he had a net worth exceeding $200 billion. On July 5, 2021, Bezos stepped down as the CEO and president of Amazon and took over the role of executive chairman. Amazon Web Services CEO Andy Jassy succeeded Bezos as the CEO and president of Amazon.
# """

def ice_break_with(model_name:str, name: str) -> Summary:
    linkedin_username= lookup(name=name)

    print('\nEntering Summarizer mode.....\n')
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url="random", mock=True)

    summary_template = """
       Given the LinkedIn information {information} about a person, I want you to create:
       1. A short summary 
       2. Two interesting facts about them
       3. Do not hallucinate any information outside of what is given
       Remember your output just display the information asked and nothing else.
       \n{format_instructions}
       """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={"format_instructions": summary_parser.get_format_instructions()},
    )

    llm = ChatOllama(temperature=0, model=model_name, num_gpu=1)
    chain = summary_prompt_template | llm | summary_parser

    response:Summary = chain.invoke({"information": linkedin_data})

    print(f"About {linkedin_username}:\n",response)

if __name__ == "__main__":
    start_time = time.time()


    # model_name = "llama3.2"
    model_name = "qwen2.5:14b"

    print("Ice Breaker Enter")
    ice_break_with(model_name, name="Eden Marco")

    end_time = time.time()

    # Calculate elapsed time
    hours, minutes, seconds = calculate_elapsed_time(start_time, end_time)

    print(
        f"\nTime taken: {hours} hours, {minutes} minutes, {seconds} seconds")