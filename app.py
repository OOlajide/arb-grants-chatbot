import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

st.title("Arbitrum Grants Chatbot")

with st.expander("About"):
    st.write("Chat with a dataset of all Arbitrum grants as seen on [Karma](https://gap.karmahq.xyz/arbitrum).")
    URL_STRING = "https://github.com/OOlajide/arb-grants-chatbot/tree/main"
    st.markdown(
        f'<a href="{URL_STRING}" style="display: inline-block; padding: 12px 20px; background-color: #4CAF50; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">Action Text on Button</a>',
        unsafe_allow_html=True
    )
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
      
