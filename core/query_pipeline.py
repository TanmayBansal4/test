from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from .tech_query_test import process_tech_query
import json

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

router_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are an intelligent routing assistant. Your task is to classify the user's query as one of the following intents:

- GENERAL: Greetings, chitchat, small talk, out-of-domain, irrelevant, or casual questions.
- TECHNICAL: Queries that are related to any labour law and likely need technical documentation or expert knowledge.

Respond strictly in this JSON format:
{{"intent": "<intent_type>"}}

Query: {query}
"""
)

import json

def route_query(llm, query: str) -> str:
    chain = router_prompt | llm
    result = chain.invoke({"query": query})

    try:
        data = json.loads(result.content.strip())
        return data.get("intent", "GENERAL")
    except json.JSONDecodeError:
        # Fail-safe fallback
        return "GENERAL"

def process_query(query, state, perspective_name, chat_history, session_id):
    print(chat_history)
    # Route the query
    intent = route_query(llm, f"{query} for {state}")

    print("🔀 Routed intent:", intent)

    if intent == "GENERAL":
        response = llm.invoke(
            f"""
If the user greets, reply with a polite greeting.
If the query is casual, irrelevant, or outside labour laws,
politely decline.

User query:
{query}
"""
        )
        return response.content

    elif intent == "TECHNICAL":
        # if state == "Central":
        #     if query == "":
        #         print("### 1. The Rule (From Context)\nUnder the Code on Wages, 2019, an employer is permitted to deduct wages from an employee's pay for absences from work. Specifically, Section 20(1) states that deductions can only be made for the absence of an employee from the place where they are required to work, as per the terms of their employment (the_code_on_wages_2019_no._29_of_2019, Page 11). The amount deducted for absence cannot exceed the proportion of wages payable to the employee based on the duration of their absence relative to the total period they were required to work during that wage period (the_code_on_wages_2019_no._29_of_2019, Page 11). Furthermore, if ten or more employees act in concert and absent themselves without due notice, the employer may deduct an amount not exceeding the wages for eight days from any such employee (the_code_on_wages_2019_no._29_of_2019, Page 12).\n\n### 2. Legal Definition (If Required)\nThe term \"wages\" is defined under the Code on Wages, 2019, as all remuneration, by whatever name called, which would, if the terms of the contract of employment were fulfilled, be payable to a person in respect of their employment (As per the Central Code, Section 2(yy)).\n\n### 3. Old vs New Analysis (CRITICAL)\nThe provisions regarding wage deductions for absence under the Code on Wages, 2019, represent a significant shift from previous legislation. Under the Factories Act, 1948, deductions for absence were less clearly defined, and the criteria for such deductions were not as stringent (Factories Act, 1948). The new Code provides a clearer framework that mandates proportional deductions based on the duration of absence, which was not explicitly required in earlier laws (the_code_on_wages_2019_no._29_of_2019, Page 11). \n\nAdditionally, the previous Contract Labour (Regulation and Abolition) Act, 1970, did not address wage deductions for absence in detail, whereas the new Code explicitly allows deductions only for authorized reasons, including absence, and sets limits on the amount that can be deducted (the_code_on_wages_2019_no._29_of_2019, Page 12). \n\nIn summary, the Code on Wages, 2019, consolidates and clarifies the rules regarding wage deductions for absence, providing a more structured approach compared to the older Acts, which lacked specific provisions on this matter.")
        #     if query == "":
        #         print("")
        # else: 
        return process_tech_query(query, state, perspective_name, chat_history)

    return "Unable to determine query intent."

