import os
import os.path
# Import necessary modules for text splitting, document loading, vector storage, and embeddings
from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
import streamlit as st
# from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# OpenAI Client
from openai import OpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Set the OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = "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"
load_dotenv()
print(os.environ.get("OPENAI_API_KEY")) #key should now be available
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize OpenAI embeddings for document vectorization
embeddings = OpenAIEmbeddings()

# Check if the FAISS index files exist
index_exists = os.path.exists("faiss_index/index.faiss") and os.path.exists("faiss_index/index.pkl")

if not index_exists:
    print("Embeddings not found, creating vector storage")

    # Load documents from a PDF file
    loader = PyPDFLoader("Test1.pdf", extract_images=True)
    documents = loader.load()

    # Split the loaded documents into chunks for processing
    text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=800)
    docs = text_splitter.split_documents(documents)

    # Create a FAISS vector store from the documents and their embeddings
    db = FAISS.from_documents(docs, embeddings)
    
    # Save the FAISS index locally for later use
    db.save_local("faiss_index")

# Define a query for similarity search
query = "What is a scheduler?"

# Load the saved FAISS index and perform a similarity search with the given query
persisted_db = FAISS.load_local("faiss_index", embeddings)
docs = persisted_db.similarity_search(query=query, k=4)

content = ""
# Print the page content of the most similar document found
counter = 1
data = ""
for doc in docs:
    print("")
    print("# Context " + str(counter))
    print(doc.page_content)
    counter+=1
    content = content+doc.page_content


# Define the retriever from the vector database
retriever = persisted_db.as_retriever()
turbo_llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo'
)
# Initialize the QA chain
qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm,
                                       chain_type="stuff",
                                       retriever=retriever,
                                       return_source_documents=True
)

# Execute the query and process the response
llm_response = qa_chain(query)
print(llm_response['result'])

# Use RetrievalQA chain with GPT-3.5-turbo model
# qa = RetrievalQA.from_chain_type(llm=OpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", retriever=persisted_db.as_retriever())
# qa = RetrievalQA.from_llm(llm="gpt-3.5-turbo"), chain_type="stuff", retriever=persisted_db.as_retriever())
# result = qa.run(query)
# print (result)

# # Prompt template
# title_template = PromptTemplate(
#     input_variables= ['content', 'query'],
#     template='''Task: Create a detailed markdown structured college notes for Students
#                 Content: Create markdown notes based on the content provided 
#                 Style: Academic
#                 Tone: Professional
#                 Audience: University graduate students
#                 Word Count: Same as the amount of content you get
#                 Format: Markdown.

#                 IMPORTANT, Use all the content mentioned in the pdf, and create structured notes. Keep it reference that you are generating detailed notes for the query "{query}".

#                 Here is the content:
#                 ```
#                 {content}
#                 ```

#                 Use the above content to create detailed learning notes for university students. Here are the instructions to create the notes:
#                 1. Start the Content with heading 1 (#), and slowly as the notes progess and use other headings as well.
#                 2. Make proper use of **bolds** and italics *italics* in the notes to highlight important concepts.
#                 3. Always start the notes with proper introduction paragraph and end the notes with a proper conclusion of everything discussed.
#                 4. Keep the notes extremely detailed and long.
#                 '''
# )

# # LLMs
# llm = OpenAI(temperature=0.9) 
# title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)

# # Display input if there's a prompt
# if query and content:
#     response = title_chain.invoke({"content": content, "query": query})
#     print(response['content'])