import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests
import google.generativeai as genai


genai.configure(api_key="AIzaSyCcsCCNncteJzxLnn2VUgu05w0Iix_MLgc")  # replace with your key


def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    embedding = r.json()["embeddings"]
    return embedding


def inference(prompt):
    model = genai.GenerativeModel("gemini-2.5-pro") 
    response = model.generate_content(prompt)
    return response.text  # returns plain string


df = joblib.load('embeddings.joblib')


incoming_query = input("Ask a Question: ")
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
to minutes) and guide the user to go to that particular video. If User asks unrelated 
question, tell him that you can only answer questions related to the course.
'''

with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)


response = inference(prompt)
print(response)


with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)

























# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import joblib
# import requests
# import google.generativeai as genai

# genai.configure(api_key="AIzaSyCcsCCNncteJzxLnn2VUgu05w0Iix_MLgc")

# # def create_embedding(text_list):
# #     embedding_model = "models/embedding-001"
# #     embeddings = []
# #     for text in text_list:
# #         result = genai.embed_content(
# #             model=embedding_model,
# #             content=text
# #         )
# #         embeddings.append(result["embedding"])
# #     return embeddings

# def create_embedding(text_list):
#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": "bge-m3",
#         "input" : text_list
#     })

#     embedding = r.json()["embeddings"]
#     return embedding


# # def inference(prompt):
# #     r = requests.post("http://localhost:11434/api/generate", json={
# #         # "model": "deepseek-r1"
# #         "model": "llama3.2",
# #         "prompt": prompt,
# #         "stream": False
# #     })

# #     response = r.json()
# #     print(response)
# #     return response

# def inference(prompt):
#     model = genai.GenerativeModel("gemini-2.5-pro")  
#     response = model.generate_content(prompt)
#     return response.text

# df = joblib.load('embeddings.joblib')


# incoming_query = input("Ask a Question: ")
# question_embedding = create_embedding([incoming_query])[0]
# # print(question_embedding)


# similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# # print(similarities)
# top_results = 5
# max_indx = similarities.argsort()[::-1][0:top_results]
# # print(max_indx)
# new_df = df.loc[max_indx]
# # print(new_df[["title","number","text"]])


# prompt = f'''I am teaching web development in my Sigma web devlopment course. here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

# {new_df[["title", "number","start","end","text"]].to_json(orient="records")}
# ----------------------------------------------
# "{incoming_query}"

# User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, it's just for you) where and how much content is taught in which video (in which video and at what timestamp, convert the timestamp from seconds to minutes) and guide the user to go to that particular video. If User asks unrelated question, tell him that you can only answer questions related to the course 
# '''

# with open("prompt.txt","w") as f:
#     f.write(prompt)


# response = inference(prompt)["response"]
# print(response)

# with open("response.txt","w") as f:
#     f.write(response)

# # for index, item in new_df.iterrows():
# #     print(index,item["title"], item["number"], item["text"], item["start"], item["end"])