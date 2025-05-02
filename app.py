import streamlit as st
import pandasai as pai
import pandas as pd
import matplotlib.pyplot as plt
import os

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
  st.write("Chat with a dataset of Arbitrum grants scraped from [Karma GAP](https://gap.karmahq.xyz/arbitrum).")
  st.write("TIP: if you are having trouble getting the information you need, try refining your question using one or more of the dataset column names: `grantee`, `grant_date`, `grant_amount_arb`, `grant_name`, `proposal_url`, `gap_url`.")
  st.write("TIP: In addition to asking questions, you can request visualizations — for example, 'Plot a bar chart of the top 10 grantees by amount received.'")
  st.write("All amounts are denominated in ARB.")

pai.api_key.set(st.secrets.pai_api_key)
df = pai.read_csv("arbitrum_grantees.csv")

with st.form("Question"):
  question = st.text_area("Question", value="What are the top 5 grantees by amount received, and how much did they receive?")
  submitted = st.form_submit_button("Submit")
  if submitted:
    with st.spinner("Thinking..."):
      try:
        response = df.chat(question)
      except Exception as e:
        st.error(f"Error: {e}. Refine your question and try again.")
        response = None
      image_dir = os.path.join("exports", "charts")
      png_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
      if png_files:
        image_path = os.path.join(image_dir, png_files[0])
        im = plt.imread(image_path)
        st.image(im, width=800)
        os.remove(image_path)
      elif response is not None:
          st.write(response)

with st.expander("View dataset"):
  st.dataframe(df)

st.download_button(
  label="Download dataset as CSV",
  data=df.to_csv().encode('utf-8'),
  file_name='arbitrum_grantees.csv',
  mime='text/csv',
)
