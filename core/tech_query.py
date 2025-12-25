def process_tech_query(query, state, perspective_name, chat_history): 
    import os
    import re
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
    from langchain_core.prompts import PromptTemplate
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    import os
    from pathlib import Path
    from dotenv import load_dotenv

    INDEX_FOLDER_DIC = {
        "Maharashtra" : "maha_ada",
        "Gujarat" : "guj_ada",
        "Uttrakhand" : "uk_ada",
        "Central" : "cen_ada",
        "Jharkhand" : "jha_ada",
        "Karnataka" : "ka_ada",
        "Uttar Pradesh" : "up_ada",
    }

    # ===================== LOAD ENV =====================
    ROOT_DIR = Path(__file__).resolve().parents[2]
    load_dotenv(ROOT_DIR / ".env")
    EMBEDDING_DEPLOYMENT_NAME = "text-embedding-ada-002"
    API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

    if not API_KEY or not AZURE_ENDPOINT or not CHAT_DEPLOYMENT:
        raise RuntimeError("Azure OpenAI config missing. Check .env location.")

    INDEX_FOLDER = f"core/unified_index_state/{INDEX_FOLDER_DIC[state]}"

# ======================================================
# UTILS
# ======================================================
    def safe_text(text: str) -> str:
        """Remove bad unicode/control characters"""
        text = text.encode("utf-8", errors="ignore").decode("utf-8")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def format_docs_with_citation(docs):
        blocks = []
        for d in docs:
            source = d.metadata.get("source", "Unknown")
            page = d.metadata.get("page", "?")
            content = safe_text(d.page_content)
            blocks.append(
                f"[SOURCE: {source} | PAGE: {page}]\n{content}"
            )
        return "\n\n".join(blocks)

# ======================================================
# MAIN
# ======================================================
    print("🔹 Loading embeddings (local)...")
    embeddings = AzureOpenAIEmbeddings(azure_deployment=EMBEDDING_DEPLOYMENT_NAME, openai_api_version=API_VERSION)
    # vector_store = FAISS.load_local(INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)

    print("🔹 Loading FAISS index...")
    vector_store = FAISS.load_local(
        INDEX_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("🔹 Initializing Azure OpenAI (safe mode)...")
    llm = AzureChatOpenAI(
        azure_deployment=CHAT_DEPLOYMENT,
        azure_endpoint=AZURE_ENDPOINT,
        openai_api_version=API_VERSION,
        api_key=API_KEY,
        temperature=0
    )

    prompt_template =  """You are a Senior Legal Analyst specializing in Indian Labour Reforms, including:
- Code on Wages
- Occupational Safety, Health and Working Conditions Code (OSHWC)
- Code on Social Security
- Industrial Relations Code

Your analysis must be legally precise, citation-driven, and jurisdiction-aware.

------------------------------------------------------------
ANALYTICAL PERSPECTIVE:
------------------------------------------------------------
You must analyze the query strictly from the following legal perspective:
{perspective_name}

- The perspective defines the PRIMARY labour code or legal lens to apply.
- Do NOT introduce other labour codes unless they are legally necessary for comparison.
- If the query falls outside this perspective, clearly state so.

------------------------------------------------------------
CHAT HISTORY (CONTEXT ONLY – DO NOT CITE):
------------------------------------------------------------
The following is the prior conversation for contextual understanding ONLY.
- Use it to understand intent, continuity, and follow-up nature.
- DO NOT treat chat history as a legal source.
- DO NOT cite chat history.
- ALL legal conclusions must come from CONTEXT or Central Code references.

{chat_history}

------------------------------------------------------------
INSTRUCTIONS (STRICT):
------------------------------------------------------------

1. **Source Material Constraint**
   - You MUST primarily rely on the provided CONTEXT.
   - The CONTEXT is extracted from State Draft Rules and/or Central Labour Codes.
   - The CONTEXT contains explicit [SOURCE] and [PAGE] tags.

2. **Citation Rule (MANDATORY)**
   - EVERY factual or legal statement MUST end with a citation in the format:
     (File Name, Page No)
   - Example:
     "The employer must maintain electronic registers (OSHWC_Rules.pdf, Page 42)."

3. **Use of Internal Knowledge (Controlled)**
   - If the CONTEXT does NOT define a required legal term or background:
     - You MAY use internal knowledge of the relevant Central Labour Code.
     - You MUST explicitly state:
       "As per the Central Code (Internal Legal Reference)..."
     - Mention the exact Section number.
     - Clearly distinguish internal legal reference from contextual facts.

4. **Language**
   - Answer strictly in **ENGLISH**.
   - Do NOT translate statutory text unless necessary for explanation.

5. **No Hallucination Rule**
   - If the answer is NOT found in the CONTEXT and cannot be reasonably supplemented by Central Code knowledge:
     - Clearly state:
       "This information is not found in the provided documents."

------------------------------------------------------------
REQUIRED RESPONSE STRUCTURE:
------------------------------------------------------------

### 1. The Rule (From Context)
- Explain what the provided State Draft Rules or Central Code say about the issue.
- Some content in the CONTEXT may be irrelevant; you must identify and use only what is legally relevant.
- Mention the specific Rule number, Section number, Form number, or procedural reference where available.
- EACH sentence MUST include a citation (File Name, Page No).

### 2. Legal Definition (If Required)
- If the CONTEXT does not define a key legal term:
  - Provide the definition using Central Labour Code knowledge.
  - Explicitly label it as:
    "As per the Central Code (Internal Legal Reference)"
  - Mention the applicable Section number.

### 3. Old vs New Analysis (CRITICAL)
- Compare the provision with corresponding older legislation such as:
  - Factories Act, 1948
  - Contract Labour (Regulation and Abolition) Act, 1970
  - Inter-State Migrant Workmen Act, 1979
- Clearly state:
  - What has changed
  - What is newly introduced
  - What has been removed, merged, or consolidated
- If there is no substantive change, explicitly state:
  "This provision remains largely similar to the previous Act."

------------------------------------------------------------
CONTEXT:
------------------------------------------------------------
{context}

------------------------------------------------------------
QUESTION:
------------------------------------------------------------
{question}

------------------------------------------------------------
DETAILED ANALYST RESPONSE (IN ENGLISH):
------------------------------------------------------------

"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "perspective_name", "chat_history"]
    )

    query = query
    print(f"\n❓ QUERY: {query}")

    def expand_query(query: str) -> str:
        prompt = PromptTemplate(
            input_variables=["query"],
            template="""Given the following query, generate 10 related phrases that could be relevant to the query.
            Query: {query}
            Related terms:"""
        )
        chain = prompt | llm
        response = chain.invoke(query)
        expanded_terms = [term.strip() for term in response.content.split(',')]
        expanded_query = f"{query} {' '.join(expanded_terms)}"
        return expanded_query

    query = expand_query(query)
    # print(query)

    retriever = vector_store.as_retriever(search_kwargs={"k": 12})
    docs = retriever.invoke(query)

    if not docs:
        print("❌ No documents retrieved.")
        return

    context = format_docs_with_citation(docs)

    # print(context)

    final_prompt = PROMPT.format(
    context=context,
    question=safe_text(query),
    perspective_name=perspective_name,
    chat_history=chat_history or "No prior conversation."
)


    print("\n🤖 Generating answer...")
    response = llm.invoke(final_prompt)

    print("\n📢 ANSWER:")
    return response.content

