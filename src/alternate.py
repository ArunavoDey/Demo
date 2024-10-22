import requests
from bs4 import BeautifulSoup
import requests, lxml
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain import hub
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
link_array = []
def search_query(query, num_results=100):
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
            idx_2 = 0
            for result in soup.select('.tF2Cxc'):
                title = result.select_one('.DKV0Md').text
                link = result.select_one('.yuRUbf a')['href']
                link_array.append(link)
                print(title, link, sep='\n')
                idx_2 = idx_2 + 1
            print(f"there are {idx_2} results in {idx} response")
            print(link_array)    
            #print(f"printing len for {idx} {len(link_array)}")
            starts = starts+ idx_2
            idx = idx + 1
        else:
            print(f"Error: Unable to fetch search results (status code: {response.status_code})")
    return link_array
if __name__ == "__main__":
    # Ask for user input
    query = input("Enter your search query: ")
    #num_results = int(input("Enter the number of results to return: "))

    # Get the URLs from the search results
    results = search_query(query)
    os.environ["OPENAI_API_KEY"] = input("Enter passphrase for OpenAPI:")

    print(results)
    llm = ChatOpenAI(model="gpt-4o-mini")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs[0:10])
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context":retriever | format_docs, "question":RunnablePassthrough()}
        | prompt
        |llm
        |StrOutputParser()
    )
    rag_chain.invoke(query)
    vectorstore.delete_collection()

