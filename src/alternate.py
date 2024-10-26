import os
import copy
import requests
from bs4 import BeautifulSoup
import requests, lxml
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain import hub
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
#from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import bs4
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from concurrent.futures import ThreadPoolExecutor
doc_l = []
def search_query(query, link_array, num_results=100):
    # Construct the search URL (using Google search)
    response_array = []
    starts_array = [1, 101, 1000]
    starts = 1
    idx = 1
    idx_2 = 1
    while idx_2 > 0:
        search_url = f"https://www.google.com/search?q={query}&num={num_results}&start={starts}"
        # Set the User-Agent to simulate a browser request
        headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        # Send a GET request to fetch the search page
        response = requests.get(search_url, headers=headers)
        #response_array.append(response)
        #for i in range(len(response_array)):
        print(response.status_code )
        #print(f"printing len of response array {len(response_array)}")
        #link_array = []
        #idx =0
        #for response in response_array:
        if response.status_code == 200:
            # Parse the search results page
            """
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all the <a> tags that contain search result URLs
            result_urls = []
            for link in soup.find_all('a'):
                href = link.get('href')
                if href.startswith('/url?q='):
                    # Extract the actual URL
                    url = href.split('/url?q=')[1].split('&')[0]
                    result_urls.append(url)
                
                # Break the loop if the specified number of results is found
                if len(result_urls) >= num_results:
                    break
            """
            soup = BeautifulSoup(response.text, 'lxml')
            #link_array = []
            idx_2 = 0
            for result in soup.select('.tF2Cxc'):
                title = result.select_one('.DKV0Md').text
                link = result.select_one('.yuRUbf a')['href']
                if '.pdf' in link:
                    continue
                link_array.append(link)
                print(title, link, sep='\n')
                idx_2 = idx_2 + 1
            print(f"there are {idx_2} results in {idx} response")    
            #print(f"printing len for {idx} {len(link_array)}")
            starts = starts+ idx_2
            idx = idx + 1
        else:
            print(f"Error: Unable to fetch search results (status code: {response.status_code})")
    return link_array

def load_docs(link):
    global doc_l
    loader = AsyncHtmlLoader(link)
    doc = loader.load()
    html2text = Html2TextTransformer()
    documents_transformed = html2text.transform_documents(doc)
    """
    if len(doc_l) == 0:
        doc_l = copy.deepcopy(documents_transformed)
    else:
        doc_l = doc_l + copy.deepcopy(documents_transformed)
    """
    doc_l.append(copy.deepcopy(documents_transformed))
    print(f"printing from load docs {len(doc_l)}")
    #print(len(doc_l))
    return doc
if __name__ == "__main__":
    torch.cuda.empty_cache()
    os.environ["GOOGLE_CLOUD_API_KEY"]=input("Enter passphrase for Google API:")
    os.environ["GOOGLE_API_KEY"]=input("Enter passphrase for Google API:")
    os.environ["LANGCHAIN_API_KEY"]=input("Enter passphrase for Langchain:")
    os.environ["LANGCHAIN_TRACING_V2"]="True"
    os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"]=input("Enter projectname for Langchain:")
    genai.configure(api_key=os.environ["GOOGLE_CLOUD_API_KEY"])
    query = input("Enter your query: ")#"Homicide numbers of new orleans and new york for years 2024, 2023, 2022, 2021 and 2020" #input("Enter your search query: ")
    #num_results = int(input("Enter the number of results to return: "))
    link_array =[]
    # Get the URLs from the search results
    results = search_query(query, link_array)
    n_link_array = []
    sub_array = []
    #doc_l = []
    for i in range(len(results)):
        print(results[i])
        sub_array.append(results[i])
        if i%10 == 0 or i == len(results)-1:
            n_link_array.append(copy.deepcopy(sub_array))
            sub_array.clear()

    docss = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future = {executor.submit(load_docs, result):result for result in n_link_array}
        #print(future)
        #loader = AsyncHtmlLoader(result)
        docss.append(future)
    #print(doc_l[0])
    print("printing docs")
    #print(docss[0])
    print("******************printing documents transformed***********************")
    print(len(doc_l))
    print(len(docss))
    d_transformed = []
    for doclist in doc_l:
        for i in range(len(doclist)):
            d_transformed.append(doclist[i])
    print("printing length of actual documehts transformed")
    print(len(d_transformed))
    """
    html2text = Html2TextTransformer()
    documents_transformed = html2text.transform_documents(doc_l)
    print("printing trannsformed documents content")
    print(documents_transformed[0].page_content)
    """
    #embeddings = SentenceTransformerEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code":True})
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False, 'batch_size':1}
    hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
    )

    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function= hf, #embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )
    uuids = [str(uuid4()) for _ in range(len(d_transformed))]
    vector_store.add_documents(documents=d_transformed, ids=uuids)
    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    retriever = vector_store.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context":retriever | format_docs, "question":RunnablePassthrough()}
        | prompt
        |llm
        |StrOutputParser()
    )
    r = rag_chain.invoke(query)
    print("LLM Agent response is: ")
    print(r)
    vector_store.delete_collection()
    
