import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests
import google.generativeai as genai



genai.configure(api_key="AIzaSyCcsCCNncteJzxLnn2VUgu05w0Iix_MLgc")

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model":"bge-m3",
        "input": text_list
    })
    return r.json()["embeddings"]


def inference(prompt):
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    return response.text

@st.cache_resource
def load_db():
    return joblib.load("embeddings.joblib")

df = load_db()


st.set_page_config(page_title="Web Dev Helper", layout="wide")
st.title("Wed Development Q&A")
st.write("Ask a question about the course and get the video + timestamp where it's explained.")


incoming_query = st.text_input("Ask your question:")

if st.button("Get Answer") and incoming_query:
    with st.spinner("Searching...wait for it"):
        question_embedding = create_embedding([incoming_query])[0]


        similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
        top_results = 5
        max_indx = similarities.argsort()[::-1][0:top_results]
        new_df = df.loc[max_indx]



        prompt = f'''I am teaching web development in my Sigma web devlopment course. 
        Here are video subtitle chunks containing video title, video number, start time in seconds, 
        end time in seconds, the text at that time:

        {new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}

        ----------------------------------------------
        "{incoming_query}"

        User asked this question related to the video chunks, you have to answer in a human way 
        (don't mention the above format, it's just for you) where and how much content is taught 
        in which video (in which video and at what timestamp, convert the timestamp from seconds 
        to minutes) and guide the user to go to that particular video. give the timestamps in bullet points. If User asks unrelated 
        question, tell him that you can only answer questions related to the course.
        '''

        response = inference(prompt)


        st.subheader("Answer")
        st.write(response)


        with st.expander("Matched Subtitle Chunks"):
            st.dataframe(new_df[["title","number","start","end","text"]])