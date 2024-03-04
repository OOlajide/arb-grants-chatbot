import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

st.title("Arbitrum Grants Chat")

with st.expander("About"):
    st.write("about")

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
      
