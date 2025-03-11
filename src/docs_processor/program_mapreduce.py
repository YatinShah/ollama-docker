# ref: https://vincent.codes.finance/posts/documents-llm/

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain

file_path = "./Paper/Paper More_is_Less.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# Define LLM Chain

llm = ChatOpenAI(
    temperature=0.1,
    model_name="mixtral:8x7b",
    api_key="ollama",
    base_url="http://localhost:7869/v1",
)

print(f"Loaded {len(docs)} documents")
user_query = "What is the data used in this analysis?"
map_template = """The following is a set of documents
{docs}
Based on this list of documents, please identify the information that is most relevant to the following query:
{user_query} 
If the document is not relevant, please write "not relevant".
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)
map_prompt = map_prompt.partial(user_query=user_query)
map_chain = LLMChain(llm=llm, prompt=map_prompt)

reduce_template = """The following is set of partial answers to a user query:
{docs}
Take these and distill it into a final, consolidated answer to the following query:
{user_query} 
Complete Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_prompt = reduce_prompt.partial(user_query=user_query)

from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain

reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="docs"
)

# Combines and iteratively reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    combine_documents_chain=combine_documents_chain,
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=4000,
)

# Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_documents_chain,
    document_variable_name="docs",
    return_intermediate_steps=False,
)
print(f"LLM chain is ready: {map_reduce_chain}")

result = map_reduce_chain.invoke(docs)
print(f"Result: \n\n\n{result}")