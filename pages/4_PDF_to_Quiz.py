import os
import streamlit as st
from util_py.ui_utils import check_password
from util_py.pdf_to_quizz import pdf_to_quizz
from util_py.text_to_quizz import txt_to_quizz
from util_py.generate_pdf import generate_pdf_quiz
import json

import asyncio

st.title("Quiz Master")

with st.sidebar:
    try:
        OPENAI_API_KEY = st.text_input("Enter your OpenAI API Key", type="password")
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    except ValueError as e:
        st.error(str(e))

def build_question(count, json_question):

    if json_question.get(f"question") is not None:
        st.write("Question: ", json_question.get(f"question", ""))
        choices = ['A', 'B', 'C', 'D']
        selected_answer = st.selectbox(f"Select your answer:", choices, key=f"select_{count}")
        for choice in choices:
            choice_str = json_question.get(f"{choice}", "None")
            st.write(f"{choice} {choice_str}")
                    
        color = ""
        if st.button("Submit", key=f"button_{count}"):
            rep = json_question.get(f"reponse")
            if selected_answer == rep:
                color = ":green"
                st.write(f":green[Correct answer: {rep}]")
                
            else:
                color = ":red"
                st.write(f":red[Wrong answer. The correct answer is {rep}].")                

        st.write(f"{color}[Your answer: {selected_answer}]")

        count += 1

    return count

st.markdown("### Upload Notes/Study Material for making the quiz")

# Upload PDF file
uploaded_file = st.file_uploader("Upload pdf here",type=["pdf"])
txt = st.text_area('Enter the text from which you want to generate the quiz')

if st.button("Generate Quiz", key=f"button_generer"):
    if txt is not None:
        with st.spinner("Generating quiz..."):
            st.session_state['questions'] = asyncio.run(txt_to_quizz(txt))
            st.write("Quiz generated successfully!")

if uploaded_file is not None:    
    old_file_name = st.session_state.get('uploaded_file_name', None)
    if (old_file_name != uploaded_file.name):
        # Convert PDF to text
        with st.spinner("Generating quiz..."):

            with open(f"data/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getvalue())        

            # Initialize session state
            st.session_state['uploaded_file_name'] = uploaded_file.name
            st.session_state['questions'] = asyncio.run(pdf_to_quizz(f"data/{uploaded_file.name}"))

            st.write("Quiz generated successfully!")

if ('questions' in st.session_state):
    # Display question
    count = 0
    for json_question in st.session_state['questions']:
        # Your code to display each question goes here
        count = build_question(count, json_question)


        
    # generate pdf quiz
    if st.button("Generate PDF Quiz", key=f"button_generer_quiz"):
        with st.spinner("Generating quiz in PDF..."):
            json_questions = st.session_state['questions']
                # save into a file
            file_name = uploaded_file.name

            # remove extension .pdf from file name
            if file_name.endswith(".pdf"):
                file_name = file_name[:-4]

            with open(f"data/quiz-{file_name}.json", "w", encoding='utf-8', errors='ignore') as f:
                str = json.dumps(json_questions)
                f.write(str)

        generate_pdf_quiz(f"data/quiz-{file_name}.json", json_questions)
            
        st.write("PDF Quiz generated successfully!")        
