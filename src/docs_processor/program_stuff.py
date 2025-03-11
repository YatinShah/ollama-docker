# ref: https://vincent.codes.finance/posts/documents-llm/
#  other refs: https://www.python-engineer.com/posts/langchain-crash-course/ 
#  LangChain Tuts: https://python.langchain.com/docs/tutorials/ 
#  Video: https://www.youtube.com/watch?v=8BV9TW490nQ 

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

file_path = "./Paper/Paper More_is_Less.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load()

print(f"Loaded {len(pages)} documents")

for doc in pages:
# Prompt
    docs = [doc]
    prompt_template = """Write a 100 to 150 words long summary of the following document. 
    Only include information that is part of the document. 
    Do not include your own opinion or analysis.

    Document:
    "{document}"
    Summary:"""
    prompt = PromptTemplate.from_template(prompt_template)
    print(f"Prompt is ready: {prompt}")

    # Define LLM Chain

    llm = ChatOpenAI(
        temperature=0.1,
        model_name="mixtral:8x7b",
        api_key="ollama",
        base_url="http://localhost:7869/v1",
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    print(f"LLM chain is ready: {llm_chain}")

    # Create full chain

    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="document"
    )
    print(f"Stuff chain is ready: {stuff_chain}")

    # Invoke with limited pages
    result = stuff_chain.invoke(docs)
    print(f"Result: \n\n\n{result}")