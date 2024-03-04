import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

st.set_page_config(
    page_title="Arbitrum Grants Chatbot",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get Help": None,
        "Report a bug": "https://twitter.com/sageOlamide",
        "About": None
    }
)

st.title("Arbitrum Grants Chatbot")

with st.expander("About"):
    st.write("Chat with a dataset of Arbitrum grants, scraped from [Karma GAP](https://gap.karmahq.xyz/arbitrum).")
    
# load dataset
dataset = pd.read_csv("arbitrum_grantees.csv")
# Instantiate a LLM
llm = OpenAI(api_token=st.secrets.api_key)
df = SmartDataframe(dataset, config={"llm": llm})

with st.form("Question"):
  question = st.text_input("Question", value="", type="default")
  submitted = st.form_submit_button("Submit")
  if submitted:
    with st.spinner("Thinking..."):
      response = df.chat(question)
      if response is not None:
        st.write(response)
        
with st.expander("View dataset"):
  st.dataframe(dataset)
      
st.download_button(
    label="Download dataset as CSV",
    data=dataset.to_csv().encode('utf-8'),
    file_name='arbitrum_grantees.csv',
    mime='text/csv',
)
