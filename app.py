#%%
# 0.requirement importation
# Pytorch Data Loader Object
from torch.utils.data import DataLoader
# 
from fastapi import FastAPI
# Pytorch Library
import torch
# # Type Validation
from typing import List
from pymilvus import connections, AsyncMilvusClient, Collection, MilvusClient
import numpy as np
# This class is the main interface for running offline inference with the vLLM engine. 
# It enables you to control various aspects of how the model generates text, such as randomness, token selection, and stopping criteria.
from vllm.sampling_params import SamplingParams
# images processing
import base64
# 
import aiofiles
import base64

from typing import List
# Accelerate calculations
from colpali_engine.utils.torch_utils import ListDataset
#
# import urllib.parse
import urllib.parse

from pprint import pprint
import httpx
import asyncio
import concurrent.futures
from embedding_model import model, processor
import re
import datetime
model = model
processor = processor

# pixtral docker image  api
pixtral_api = "http://localhost:8000/v1/chat/completions"



#%%__________________________________________________________________________________________
# Milvus is deployed as a standalone instance using Docker Compose.
#*** This module should be refactored to use asynchronous programming in production.***
client = MilvusClient(uri="http://localhost:19530")
connections.connect("default", host="localhost", port="19530")

# collection_name = "phase01_v2"
# collection = Collection(collection_name)
# collection.load()

# print(collection.num_entities)

#__________________________________________________________________________________________
#***This module should be refactored to use asynchronous programming in production.***

# 1. Query embedding function..
def queries_embedding(queries : list):
    # Create a DataLoader to iterate over the list of queries, processing each query
    # individually to fit model input requirements
    dataloader = DataLoader(
        dataset=ListDataset[str](queries),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: processor.process_queries(x),
    )
    qs: List[torch.Tensor] = []
    for batch_query in dataloader:
        with torch.no_grad():
            batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
            embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))
    return qs

# --------------------------------------------------------------------------------------------
# topk :  It represents the top retrieved documents (pages) from the similarity search.
topk = 10

# search_params : It represents the search parameters for the similarity search.
search_params = {"metric_type": "COSINE", "params": {}}

#__________________________________________________________________________________________
# 2. Retrieve function
def retriever(qs, collection_name, topk=topk, search_params= search_params):
  collection = Collection(collection_name)
  collection.load()
  def rerank_single_doc(doc_id, data, client, collection_name):
    # Rerank a single document by retrieving its embeddings and calculating the similarity with the query.
    doc_colbert_vecs = client.query(
        collection_name=collection_name,
        filter=f"doc_id in [{doc_id}, {doc_id + 1}]",
        output_fields=["seq_id", "vector", "doc"],
        limit=1000,
    )
    doc_vecs = np.vstack(
        [doc_colbert_vecs[i]["vector"] for i in range(len(doc_colbert_vecs))]
    )
    # score = np.dot(data, doc_vecs.T).max(1).sum()
        # dot product between the query and the document embeddings
    score = np.dot(data, doc_vecs.T).max(1).mean()
    return (score, doc_id)
    #***************************************************************************

  images_paths = []
  for query in qs:
      #0. Get all the documents that contain at least 1 similar (token-patch)
      query = query.float().numpy()
      results = client.search(
          collection_name,
          query,
          limit=100,
          output_fields=["vector", "seq_id", "doc_id"],
          search_params=search_params,
      )

      #1. Retrieve all document IDs that contain at least one similarity between the query tokens and the document patches
      doc_ids = set()
      for r_id in range(len(results)): # len(number of tokens) : for each query token
          for r in range(len(results[r_id])): # for each similar patch
              doc_ids.add(results[r_id][r]["entity"]["doc_id"]) # add the document_id to the list
      # print(doc_ids)

      #2. Get the maximum similarity score for each query across all documents :
      # Example: The maximum score for the first query with image one is 20, while the maximum score with image two is 12.
      # Therefore, image one is more similar to the query than the other images.

      #2.1. Create the similarity search function
      scores = []
      #2.2. # Run the rerank(document) task in parallel for up to 300 workers
      with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
              futures = {
                  executor.submit(
                      rerank_single_doc, doc_id, query, client, collection_name
                  ): doc_id
                  for doc_id in doc_ids
              }
              for future in concurrent.futures.as_completed(futures):
                  score, doc_id = future.result()
                  scores.append((score, doc_id))

      scores.sort(key=lambda x: x[0], reverse=True)

      if len(scores) >= topk:
            scores = scores[:topk]
      else:
            scores = scores

      for i in scores:
        image_path = collection.query(expr=f"doc_id == {i[-1]}", output_fields=["doc"], limit=1)[0]['doc']
        images_paths.append(image_path)


      # select relevant images

      return images_paths, scores
  


#%%
def get_top_paths_and_scores(images_paths, scores, threshold=0.5):
    image_score_pairs = list(zip(images_paths, scores))
    top_paths = []
    top_scores = []
    for image_path, score in image_score_pairs:
        if score[0] > threshold:
            top_paths.append(image_path)
            top_scores.append(score[0])
    return top_paths, top_scores

#%%

#__________________________________________________________________________________________
# 3. Encode image to base64 function
# Async function to encode image to base64 ()
async def encode_image_to_base64(images_path):
    print("encode_image_to_base64")
    async with aiofiles.open(images_path, "rb") as image_file:
        image_data = await image_file.read()
    return base64.b64encode(image_data).decode('utf-8')

#__________________________________________________________________________________________
async def fetch_from_vllm(url, payload, headers):
  """
  Sends an async request to the vLLM API(docker container).
  """
  async with httpx.AsyncClient(timeout=240) as client_:
    response = await client_.post(url, headers=headers, json=payload, timeout=240)
    # response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    return response.json()

#__________________________________________________________________________________________
async def display_references(images_paths):
    """ Display the image references
    that was used to answer the user's query
    using the markdown format. """
    # try:

    #     if len(images_paths) <= 3 : 
    #         images_paths = images_paths
    #     else:
    #         images_paths = images_paths[:3]


        
        # Extract the title and page number from the image path
    title_pattern = r"(?<=output_test_images_150dpi/).*?(?=_page_\d+)"
    page_pattern = r"(?<=_page_)\d+"
    references = [re.search(title_pattern, i).group(0) +
                ", p. " +
                    re.search(page_pattern, i).group(0) for i in images_paths if i]
    # Create the markdown string with a header and numbered references styled in a beautiful blue (#007BFF)
    markdown_str = "**References:** \n\n"
    for i, ref in enumerate(references, 1):
        markdown_str += f"{i}. <span>{ref}</span>\n"

    # Create the output dictionary with the key 'references'
    output = {"references": markdown_str}
    return output
    # except Exception as e:
    #     return {"references": "No references found."}


# FastAPI app
app = FastAPI() 

@app.get("/vrag/{user_message:path}/{department}")
async def vrag_generation(user_message:str, department:str):
    if department == "NOCGlobalNOC" :
        department = "AdvanceSupportBusinessGovernmentSolutions"

    print("*"*70)
    # user_message = urllib.parse.unquote(user_message)
    # delete this in the production(for debugging)
    now = datetime.datetime.now() + datetime.timedelta(hours=4)
    print("request time: ", now)

    print("\n")
    query=[user_message]
    print("query : ", query)

    print("\n")
    print("department : ", department)

    print("\n")
    qs =  queries_embedding(query)
    print("Embedding : ")
    print(qs)
    
    images_paths, scores = retriever(qs=qs, collection_name=department, topk=topk, search_params= search_params)
    print("\n")
    print("score : ")
    print(scores)

    get_top_paths_and_scores(images_paths, scores, threshold=0.5)
    top_paths, top_scores = get_top_paths_and_scores(images_paths, scores, threshold=0.5)
    print("\n")
    print("top_paths ;")
    pprint(top_paths)
    print("\n")
    print("top_scores :")
    print(top_scores)
    # Encode images
    
    encoded_images = [await encode_image_to_base64(path) for path in top_paths]
    
    # Display references
    references = await display_references(images_paths=top_paths)
    # Prepare payload for vLLM API
    headers = {"Content-Type": "application/json"}
    # A dynamically sized list that contains the user content. 
    # The length of the list is equal to the number of images 
    # and will change as the number of retrieved images changes.

    user_content =    [{"type": "text", "text": user_message}] +\
    [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in encoded_images]

    payload = {
    "model": "OpenGVLab/InternVL2_5-8B-MPO",#"mistralai/Pixtral-12B-2409",
    "messages": [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """
You are a helpful assistant capable of using provided documents (in image format) **only if they are relevant** to respond comprehensively and accurately to the user’s query. Follow these guidelines for your responses:
0. 1. If the user’s question is general and does not require any images, respond using your general knowledge.
1. If the user’s question is general and does not require the provided documents, respond using your general knowledge.
2. If the user’s question is unrelated to the provided documents, rely solely on your general knowledge to answer.
3. If you cannot answer accurately based on the provided documents or your general knowledge, respond with: "I don’t have enough information to answer this question."
4. For conversational queries such as greetings or questions about your capabilities, respond politely and explain your role as an AI assistant.
5. Check if the answer is from adjacent pages and not only one. If the answer is from adjacent pages, use all the adjacent pages to answer the question completely by merging related information. For example, the answer could be divided on page one, and also on adjacent pages two and three.

Ensure all answers are detailed and avoid truncating responses.


"""


#"You are a helpful assistant capable of utilizing provided documents (in image format) **only if they are relevant** to answer in detail and complete way the user's query and don't return cutted answers.\nHere are the guidelines for your responses:\n- If the user asks a general question that does not require the provided documents(in image format), respond with an answer based on your general knowledge.\n- If the user's question is independent of the provided context, rely solely on your general knowledge to generate a response.\n- If you cannot answer the user's question accurately using either the provided documents or your general knowledge, respond with: 'I don’t have enough information to answer this question.'\n- For conversational queries like greetings or asking about your capabilities, respond politely and explain your purpose as an AI assistant."
                }
            ]
        },
        {
            "role": "user",
            "content": user_content
            # [
            #     {"type": "text",
            #       "text": user_message},

            #     {"type": "image_url",
            #       "image_url": {"url": f"data:image/png;base64,{encoded_images[0]}"}}, 
            #     {"type": "image_url",
            #       "image_url": {"url": f"data:image/png;base64,{encoded_images[1]}"}},
            #     {"type": "image_url",
            #       "image_url": {"url": f"data:image/png;base64,{encoded_images[2]}"}},
            #                     #   {"type": "image_url", "image_url": {"url": encoded_images[2]}}  # Corrected
            #     {"type": "image_url",
            #       "image_url": {"url": f"data:image/png;base64,{encoded_images[3]}"}},
            #     {"type": "image_url",
            #       "image_url": {"url": f"data:image/png;base64,{encoded_images[4]}"}},
            #       {"type": "image_url",
            #       "image_url": {"url": f"data:image/png;base64,{encoded_images[5]}"}}, 
            #     {"type": "image_url",
            #       "image_url": {"url": f"data:image/png;base64,{encoded_images[6]}"}},
            #     {"type": "image_url",
            #       "image_url": {"url": f"data:image/png;base64,{encoded_images[7]}"}},
            #     {"type": "image_url",
            #       "image_url": {"url": f"data:image/png;base64,{encoded_images[8]}"}},
            #     {"type": "image_url",
            #       "image_url": {"url": f"data:image/png;base64,{encoded_images[9]}"}},
            # ]
        }
    ],
    "temperature": 0.2,
    # "top_p": 0.95, # top_k is not usually a parameter for LLMs. top_p is more common
    "max_tokens": 1024,
    "top_k": 2,
}
    # Send request to vLLM API

    start_time = datetime.datetime.now()
    response_data = await fetch_from_vllm(pixtral_api, payload, headers)
    end_time = datetime.datetime.now()
    print("\n")
    print("Time taken: ", end_time - start_time)
    return {'answer' : response_data['choices'][0]['message']['content'],
            'references' : references['references']}

    # return response_data
  
