import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

st.title("Arbitrum Grants Chatbot")

with st.expander("About"):
    st.write("Chat with a dataset of all Arbitrum grants as seen on [Karma](https://gap.karmahq.xyz/arbitrum).")
    
# load dataset
dataset = pd.read_csv("grants_dataset.csv")
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
    data=df.to_csv().encode('utf-8'),
    file_name='grants_dataset.csv',
    mime='text/csv',
)
