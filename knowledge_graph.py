# Neo4j Driver
import neo4j 
import os
from dotenv import load_dotenv
import openai

load_dotenv()

 
openai.api_key = os.environ["OPENAI_API_KEY"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

neo4j_driver = neo4j.GraphDatabase.driver(NEO4J_URI,
                auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# LLM and Embedding Model
from neo4j_graphrag.llm import OpenAILLM 
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.neo4j_vector import Neo4jVector

ex_llm=OpenAILLM(
    model_name="gpt-4o-mini",
    model_params={
        "response_format": {"type": "json_object"}, # use json_object formatting for best results
        "temperature": 0 # turning temperature down for more deterministic results
    }
)

# Graph Schema Setup
basic_node_labels = ["Action"]

counselling_node_labels = ["Emotions", "Counselling_Technique", "Patient_Concerns", "Agenda_Setting", 
                           "Homework_Setting", "Cognitive_Behavioural_Couple_Therapy", "Socratic Questioning"]

node_labels = basic_node_labels + counselling_node_labels

# define relationship types
rel_types = ["Solves", "Leads_To", "Comprises"]

#create text embedder
embedder = OpenAIEmbeddings()

# define prompt template
prompt_template = '''
You are a counsellor tasked with extracting information from papers
and structuring it in a property graph to inform further counselling advice.

Extract the entities (nodes) and specify their type from the following Input text.
Also extract the relationships between these nodes. The relationship direction goes from the start node to the end node.

Return result as JSON using the following format:
{{"nodes": [ {{"id": "0", "label": "the type of entity", "embedding": "0", "properties": {{"name": "name of entity" }}}}],
  "relationships": [{{"type": "TYPE_OF_RELATIONSHIP", "start_node_id": "0", "end_node_id": "1", "properties": {{"details": "Description of the relationship"}} }}] }}


- Use only the information from the Input text. Do not add any additional information.  
- If the input text is empty, return empty Json. 
- Make sure to create as many nodes and relationships as needed to offer rich medical context for further research.
- Merge nodes that have similar meanings into one node for e.g. "anger" and "angry" or "agenda setting" and "Agenda setting"
- Make sure to include the embedding of each node 

Use only the following nodes and relationships:
{schema}

Assign a unique ID (string) to each node, and reuse it to define relationships.
Do respect the source and target node types for relationship and the relationship direction.

Do not return any additional information other than the JSON in it.

Examples:
{examples}

Input text:

{text}
'''

# Knowledge Graph Builder
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

kg_builder_pdf = SimpleKGPipeline(
   llm=ex_llm,
   driver=neo4j_driver,
   text_splitter=FixedSizeSplitter(chunk_size=300, chunk_overlap=100),
   embedder=embedder,
   entities=node_labels,
   relations=rel_types,
   prompt_template=prompt_template,
   from_pdf=True
)

pdf_file_paths = ["Final One-Shot Documentation caa 180125.pdf"]

import asyncio

async def process_pdfs(pdf_file_paths, kg_builder_pdf):
    for path in pdf_file_paths:
        print(f"Processing: {path}")
        pdf_result = await kg_builder_pdf.run_async(file_path=path)
        print(f"Result: {pdf_result}")

# Entry point of the script
if __name__ == "__main__":
    # Run the async function
    asyncio.run(process_pdfs(pdf_file_paths, kg_builder_pdf))

# vector_index = Neo4jVector.from_existing_graph(

#     OpenAIEmbeddings(),
#     url=NEO4J_URI,
#     username=NEO4J_USERNAME,
#     password=NEO4J_PASSWORD,
#     index_name='technique',
#     node_label="Technique",
#     text_node_properties=['name', 'description'],
#     embedding_node_property='embedding'
# )
