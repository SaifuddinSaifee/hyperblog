import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

def process_text(text, query):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # [Replace Embedding Code with OpenAI Embeddings] Convert the chunks of text into embeddings to form a knowledge base
    # embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Create vectors and use FAISS for indexing
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    
    return knowledgeBase

with st.sidebar:
    try:
        OPENAI_API_KEY = st.text_input("Enter your OpenAI API Key", type="password")
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    except ValueError as e:
        st.error(str(e))

def main():
    st.title("Create College Notes for Students Powered by AI")
    st.markdown("### Upoload your PDF document here, and put in a specific topic to generate notes")
    st.divider()

    pdf = st.file_uploader('Upload your PDF Document', type='pdf')

    query = st.text_input('Enter topic')

    isGenerate = False

    if OPENAI_API_KEY:
        isGenerate = st.button('Generate Notes')

    if isGenerate and pdf is not None:
        pdf_reader = PdfReader(pdf)
        # Text variable will store the pdf text
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Create the knowledge base object
        knowledgeBase = process_text(text, query)

        # filter relevent data
        data_chunks = knowledgeBase.similarity_search(query=query, k=5)

        content = ""

        for data_chunk in data_chunks:
            print(data_chunk.page_content)
            content = content+data_chunk.page_content

        prompt = f'''Task: Create a detailed markdown structured college notes for Students
                Content: Create markdown notes based on the content provided 
                Style: Academic
                Tone: Professional
                Audience: University graduate students
                Word Count: Same as the amount of content you get
                Format: Markdown.

                IMPORTANT, Use all the content mentioned in the context, and create structured notes. Keep it reference that you are generating detailed notes for the query "{query}".

                Here is the content:
                ```
                {content}
                ```

                Use the above content to create detailed learning notes for university students. Here are the instructions to create the notes:
                1. Start the Content with heading 1 (#), and slowly as the notes progess and use other headings as well.
                2. Make proper use of **bolds** and italics *italics* in the notes to highlight important concepts.
                3. Always start the notes with proper introduction paragraph and end the notes with a proper conclusion of everything discussed.
                4. Keep the notes extremely detailed and long.
                '''

        print(prompt)

        if prompt:
            docs = knowledgeBase.similarity_search(prompt)
            OpenAIModel = "gpt-4-turbo"
            llm = ChatOpenAI(model=OpenAIModel, temperature=0.1)
            chain = load_qa_chain(llm, chain_type='stuff')

            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=prompt)
                print(cost)

            st.write(response)

if __name__ == '__main__':
    main()