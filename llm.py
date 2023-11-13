from model import MyGPT4ALL
from knowledgebase import MyKnowledgeBase
from knowledgebase import (
    DOCUMENT_SOURCE_DIRECTORY
)

# import all the langchain modules
from langchain.chains import RetrievalQA
from langchain.embeddings import GPT4AllEmbeddings

GPT4ALL_MODEL_NAME="orca-mini-3b-gguf2-q4_0.gguf"
GPT4ALL_MODEL_FOLDER_PATH='/home/adam/anaconda3/envs/dl/lib/python3.9/site-packages'
GPT4ALL_BACKEND='llama'
GPT4ALL_ALLOW_STREAMING=True
GPT4ALL_ALLOW_DOWNLOAD=False


llm = MyGPT4ALL(
    model_folder_path=GPT4ALL_MODEL_FOLDER_PATH,
    model_name=GPT4ALL_MODEL_NAME,
    allow_streaming=True,
    allow_download=True,
    device='gpu'
)


embeddings = GPT4AllEmbeddings()

kb = MyKnowledgeBase(
    pdf_source_folder_path=DOCUMENT_SOURCE_DIRECTORY
)

kb.initiate_document_injetion_pipeline()
# get the retriver object from the vector db 

retriever = kb.return_retriever_from_persistant_vector_db(embeddings)

qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type='stuff',
    retriever=retriever,
    return_source_documents=True, verbose=True
)


while True:
    query = input("What's on your mind: ")
    if query == 'exit':
        break
    result = qa_chain(query)
    answer, docs = result['result'], result['source_documents']

    print(answer)

    print("#"* 30, "Sources", "#"* 30)
    for document in docs:
        print("\n> SOURCE: " + document.metadata["source"] + ":")
        print(document.page_content)
    print("#"* 30, "Sources", "#"* 30)