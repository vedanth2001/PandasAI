from dotenv import load_dotenv
import os
import streamlit as st

import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
load_dotenv()

API_KEY=os.environ.get('OPENAI_API_KEY')

llm=OpenAI(api_token=API_KEY)
pandas_ai=PandasAI(llm)

st.title("Prompt Driven Analysis with PandasAI")

uploaded_file=st.file_uploader("Upload a file you want to analyse (csv only)",type=['csv'])
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    
    st.write(df.head(3))

    prompt=st.text_area("Enter your prompt:")
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response.."):
                 st.write(pandas_ai.run(df,prompt=prompt))
        else:
            st.write("Please enter a prompt!")

