import streamlit as st
from llm import llm
from graph import graph

from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

from langchain.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """
You are a relationship counsellor translating user's questions about counselling techniques
into Cypher. If the user asks about a counselling method, generate a Cypher query to 
match the appropriate tools to utilise as a counsellor.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Do not return entire nodes or embedding properties.

If no data is returned, answer as you would have without any external information.
```
Schema:
{schema}

Question:
{question}

Cypher Query:
"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt,
    allow_dangerous_requests=True
)