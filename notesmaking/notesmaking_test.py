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
from apikey import OPENAI_API_KEY

# OpenAI Client
from openai import OpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

try:
    # Set the OpenAI API key as an environment variable
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    load_dotenv()
    print(os.environ.get("OPENAI_API_KEY"))  # key should now be available
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception as e:
    print(f"Failed to set or retrieve OpenAI API key: {e}")

try:
    # Initialize OpenAI embeddings for document vectorization
    embeddings = OpenAIEmbeddings()
except Exception as e:
    print(f"Failed to initialize OpenAI embeddings: {e}")

try:
    # Check if the FAISS index files exist
    index_exists = os.path.exists("faiss_index/index.faiss") and os.path.exists(
        "faiss_index/index.pkl"
    )

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
except Exception as e:
    print(f"Failed during FAISS index creation or loading: {e}")

try:
    # Define a query for similarity search
    query = "What is a scheduler?"

    # Load the saved FAISS index and perform a similarity search with the given query
    persisted_db = FAISS.load_local("faiss_index", embeddings)
    docs = persisted_db.similarity_search(query=query, k=4)

    content = ""
    # Print the page content of the most similar document found
    counter = 1
    for doc in docs:
        print("")
        # print("# Context " + str(counter))
        # print(doc.page_content)
        content = content + "# Context " + str(counter) +"\n"+ doc.page_content
        counter += 1
    
    # print(content)
except Exception as e:
    print(f"Failed during document retrieval or similarity search: {e}")

# _____________________________________________________________LLM WORK____________________________________________________

import openai
from openai import OpenAI

client = OpenAI()
# Replace with your actual OpenAI API key
openai.api_key = "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"


# User-defined parameters
model_engine = "gpt-4"  # Adjust for desired capability and cost
max_tokens = 7000  # Maximum number of tokens per response
temperature = 0.5  # Controls randomness (0: predictable, 1: creative)
top_p = 0.9  # Controls sampling bias (0: uniform, 1: focus on high probability)


def generate_response(
    prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p
):
    """Generates a creative response to the given prompt using the GPT OpenAI API.

    Args:
        prompt: The initial text or prompt to guide the generation.
        max_tokens: The maximum number of tokens in the generated response.
        temperature: Controls the randomness of the generated text (higher = more creative).
        top_p: Controls the sampling bias of the generated text (higher = focuses on high-probability tokens).

    Returns:
        The generated response text.
    """

    response = client.chat.completions.create(
        model=model_engine,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    response_message = response.choices[0].message.content
    return response_message


# User interaction loop

print("\nWelcome to the Notes writing Assistant!")
prompt = f"""Task: Create a detailed markdown structured college notes for Students
        Content: Create markdown notes based on the content provided 
        Style: Academic
        Tone: Professional
        Audience: University graduate students
        Word Count: Same as the amount of content you get
        Format: Markdown.
        IMPORTANT, Use all the content mentioned in the pdf, and create structured notes. Keep it reference that you are generating detailed notes for the query "{query}".
        Here is the content:
        ```
        {content}
        ```
        Use the above content to create detailed learning notes for university students. Here are the instructions to create the notes:
        1. Start the Content with heading 1 (#), and slowly as the notes progess and use other headings as well.
        2. Make proper use of **bolds** and italics *italics* in the notes to highlight important concepts.
        3. Always start the notes with proper introduction paragraph and end the notes with a proper conclusion of everything discussed.
        4. Keep the notes extremely detailed and long.
        5. Finally End the response with 3 questions derived from the content.
        """

# print(prompt)
response = generate_response(prompt)
print(f"{response}")
# Optional: Offer additional features like style options, genre suggestions, etc.
