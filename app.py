import os
import random

import streamlit as st
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback # Get price of each API call

from apikey import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# App framkework variable | Taking input from the user
st.title('Create your blog posts with Al')
st.write ("Introduce information about your business and the goal Of the campaign and we'll take care Of the rest!")

st.write("##")

col1, col2 = st.columns(2)

with col1:

    campaign_goal = st.radio(
        "Campaign Goal",
        ('Convince to buy product', 'Recover churned customers', 'Teach a new concept', 'Onboard new users', 'Share product updates', 'Inform/Brief'))

with col2:
    brand_tone = st.radio(
        "Brand Tone",
        ('Formal', 'Informal')
    )

    industry = st.selectbox(
        "Industry",
        ("Marketing", "Technical Consultancy", "Creative", "Event Management", "Recruitment", "Security", "Real Estate", "Law", "E-commerce", "Educational"))

st.write("##")

prompt = st.text_area(
    label = '**_Tell us about more about the Blog post you want to send, or any insturction you might want to add_**',
    height = 200,
    max_chars= 1000,
    placeholder="Innovation which revolutionzes the marketing campaigns..."
    )

st.write("##")

# rate_dict = {
#     "Medium: 700 ~ 1500" : 400,
#     "large: 1500 ~ 2000" : 800,
#     "Super Large: 2000 ~ 3000" : 4000,
# }

# char_limit = st.select_slider(
#     "Blog post length in **_characters_**",
#     ("Medium: 700 ~ 1500", "large: 1500 ~ 2000", "Super Large: 2000 ~ 3000"), value=("Medium: 700 ~ 1500")
# )
# char_limit = rate_dict[char_limit]
# words_limit = int(char_limit*0.)

# The LLMs function that generate the response based on the prompt and the supplied temperature
def llm_response(temperature):
    llm = ChatOpenAI(temperature=temperature, model='gpt-3.5-turbo', max_tokens=2000, streaming=True)
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
    response = title_chain.run(brand_tone=brand_tone, campaign_goal=campaign_goal, industry=industry, prompt=prompt)
    return response

# The generate function that calls llm_response() function to generate Blog posts
def generate_draft():
    # with get_openai_callback() as cb: # Uncomment this line of code to the total cost incurred for generating each set of 5 drafts

    st.write(llm_response(round(random.uniform(1, 1), 2)))

# Blog post generation function
if st.button('Generate'):

    title_template = PromptTemplate(    
        input_variables= ['brand_tone', 'campaign_goal', 'industry', 'prompt'],
        template="Your objective is to create a persuasive {brand_tone} blog post aimed at readers within the {industry} industry interest, as part of the {campaign_goal}. Utilize the provided topic {prompt} for context. It is essential to maintain the {brand_tone}. Do not use bullet points or numbered points in the entire blog, write a lengthy blog.To structure your blog post effectively, begin with a concise title describing the topic. Next, plan and outline the key sections, headings, and subheadings, followed by a captivating introduction that introduces the post's content. In the body, use clear headings, brief paragraphs, and incorporate lists, visuals, and hyperlinks for clarity and engagement. Finally, conclude the blog post with a friendly signoff, summarizing the main points and reinforcing the key message for the readers to remember. IMPORTANT: WRITE THE ENTIRE BLOG IN MARKDOWN ONLY, AND CREATE A TABLE OF CONTENT AFTER THE TITLE, AND THEN WRITE THE BLOG CONTENT."
        )

    generate_draft()

# Footer code
footer="""<style>
a:link , a:visited{
color: #789a00;
background-color: transparent;
text-decoration: none;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ☕ by <a style='display: block; text-align: center;' href="https://twitter.com/SaifSaifee_dev" target="_blank">Saifuddin Saifee</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)