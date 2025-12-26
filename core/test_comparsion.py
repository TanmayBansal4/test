from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
import json


states_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are an intelligent extraction assistant. Your task is to extract Indian state names and Central from the user's query
Respond strictly in this JSON format:
{{"states": ["Delhi", "Maharashtra", "Jharkhand"]}}

Query: {query}
"""
)


# ===================== LOAD ENV =====================
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

if not API_KEY or not AZURE_ENDPOINT or not CHAT_DEPLOYMENT:
    raise RuntimeError("Azure OpenAI config missing. Check .env location.")

# ===================== LLM =====================
llm = AzureChatOpenAI(
    azure_deployment=CHAT_DEPLOYMENT,
    api_key=API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    openai_api_version=API_VERSION,
    temperature=0
)

def states_query(llm, query: str) -> str:
    chain = states_prompt | llm
    result = chain.invoke({"query": query})
    data = json.loads(result.content)
    states = data["states"]
    print(len(states))

    num = len(states)
    while num > 0:
        print(states[num - 1])
        num = num - 1

    return states

