import numpy as np
import anthropic
from openai import AzureOpenAI, OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, kmeans_plusplus, MiniBatchKMeans
from sklearn_extra.cluster import KMedoids
#from InstructorEmbedding import INSTRUCTOR


from tqdm import tqdm

client_embeddings = OpenAI(
                    api_key = OPENAI_KEY
                )

def llm_api (prompt, assistant_prompt, model = "llama3.3-70b", max_completion_tokens =1000):
    print(model)

    if model == "gpt-4o" or model == 'gpt-3.5-turbo':
        # OpenAI API Key (set your own key here)
        client = OpenAI(
                    api_key = OPENAI_KEY
                )
        response = client.chat.completions.create(
        messages=[
                {"role": "system", "content": "You are a helpful assistant summarizing text clusters."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_prompt},
            ],
            max_tokens=1000,
            temperature=0,
            model=model
        )
        response_text = response.choices[0].message.content
        print(response_text)
        
    elif model == "llama3.3-70b":

        client = OpenAI(
            api_key=LLAMA_KEY,
            base_url="https://api.llama-api.com/"
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant summarizing text clusters."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_prompt},
            ],
            temperature=0,
            max_completion_tokens=max_completion_tokens,
            top_p=1,
            stream=False,
            stop=None,
            seed = 1
        )

        response_text = response.choices[0].message.content

    # Deepseek-chat
    elif model == "deepseek-chat":

        client = OpenAI( 
            api_key=DEEPSEEK_KEY,
            base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant summarizing text clusters."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_prompt}
            ],
            stream=False,
            temperature=0,
            max_tokens=max_completion_tokens,
            top_p=1,
            stop=None,
            seed = 1
        )
        if response is None:
            response_text = ""
            print(response)
            # raise Exception("Response is None")
        else:
            response_text = response.choices[0].message.content


    elif model == "claude-3-7-sonnet-20250219":
        client = anthropic.Anthropic(
            api_key=CLAUDE_KEY
        )

        message = client.messages.create(
            model=model,
            max_tokens=max_completion_tokens,
            temperature=0,
            system="You are a helpful assistant summarizing text clusters.",
            messages=[
                {"role": "user", "content":  prompt},
                {"role": "assistant", "content": assistant_prompt}
            ]
        )
        response_text = message.content[0].text


    else:
        response_text = "Model not found."




    return response_text
  
def get_embeddings(texts, model = "text-embedding-3-small", emb_type = 'openai', instructor_prompt = ""):
    """
    Get embeddings using OpenAI's text-embedding-3-small.
    """

    def inner_get_embedding(text):

        if (len(text) == 0):
            return []

        try:
            
            result = client_embeddings.embeddings.create(
                model=model,
                input=text
            )        
            return result.data
        
        except Exception as e1:
            
            if (len(text) == 0) or (len(text) == 1):
                return []
            
            text_A = text[:int(len(text)/2)]
            text_B = text[int(len(text)/2):]

            result_A = inner_get_embedding(text_A)
            result_B = inner_get_embedding(text_B)

            return result_A + result_B 

    if emb_type == 'openai':
        result = inner_get_embedding(texts)
        embeddings = [i.embedding for i in result]
        return np.array(embeddings)
    
    elif emb_type == 'distilbert':
        model = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
        embeddings = model.encode(texts)
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return np.array(normalized_embeddings)

    elif emb_type == 'e5-large':
        texts = ["query: " + text for text in texts]
        model = SentenceTransformer("intfloat/e5-large")
        embeddings = model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings)
    
    elif emb_type == 'sbert':
        model = SentenceTransformer("all-mpnet-base-v2")
        embeddings = model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings)
    
    elif emb_type == 'instructor':
        texts = [[instructor_prompt, text] for text in texts]
        model = INSTRUCTOR("hkunlp/instructor-xl")
        embeddings = model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings)

def summarize_cluster(texts, prompt = "", text_type = "", model = "gpt-4o"):
    """
    Use an LLM to generate a summary of a cluster.
    """
    if prompt == "":
        prompt = f"Write a single sentence that represents the following cluster concisely:\n\n" + "\n".join(texts) 
    else:
        prompt = prompt + "\n\n" + "\n".join(texts)  
    
    if text_type == "":
        text_type = "Sentence:"

    messages = [
        {"role": "system", "content": "You are a helpful assistant summarizing text clusters."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": text_type}
    ]
    
    response_text = llm_api(prompt,text_type,model)
    return response_text

def representative_cluster(texts,model="gpt-4o"):
    """
    Use an LLM to generate a summary of a cluster.
    """
    prompt = f"Identify the text that best represents the overall meaning and key themes of the following cluster of texts:\n\n" + "\n".join(texts)  # Limit to 5 for brevity
    
    text_type = "Representative Text:"
    messages = [
        {"role": "system", "content": "You are a helpful assistant selecting the representatives of text clusters."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": text_type}
    ]
    
    response_text = llm_api(prompt,text_type,model)
    return response_text

def sequentialMiniBatchKmeans(text_features, num_clusters, random_state, max_batch_size, max_iter = 100):

    num_batches = int(np.ceil(len(text_features)/max_batch_size))
    k, m = divmod(len(text_features), num_batches)

    #calculate batches
    text_features = [text_features[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_batches)]

    miniBatchKMeans = MiniBatchKMeans(n_clusters=num_clusters,
                         random_state=random_state,
                         batch_size=max_batch_size,
                         init="k-means++")
    
    #process each batch sequentially
    for cur_text_batch in text_features:
        miniBatchKMeans = miniBatchKMeans.partial_fit(cur_text_batch)
    
    return miniBatchKMeans
    
def miniBatchKLLMeans(text_data, 
              num_clusters,
              max_batch_size = 5000, 
              init = 'k-means++',
              prompt = "", text_type = "",
              force_context_length = 0, max_llm_iter = 5, 
              max_iter = 100, tol=1e-4, random_state = None, 
              emb_type = 'openai', text_features = None,
              final_iter = True,
              initial_iter = True,
              model = "gpt-4o"):
    
    num_batches = int(np.ceil(len(text_data)/max_batch_size))
    k, m = divmod(len(text_data), num_batches)

    #calculate batches
    text_data = [text_data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_batches)]
    text_features = [text_features[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_batches)]

    #initialize centroids and no data processed
    centroids = init
    ndata = 0
    summaries = []

    #process each batch sequentially
    for ibatch in range(num_batches):
        
        batch_assignments, batch_summaries, _, batch_centroids, _, _ = kLLMmeans(text_data[ibatch], 
              num_clusters,
              init = centroids,
              prompt = prompt, text_type = text_type,
              force_context_length = force_context_length, max_llm_iter = max_llm_iter, 
              max_iter = max_iter, tol=tol, random_state = random_state, 
              emb_type = emb_type, text_features = text_features[ibatch],
              final_iter = final_iter,
              initial_iter = initial_iter,
              model = model)
        
        batch_counts = np.array([np.sum(np.array(batch_assignments)==i) for i in range(num_clusters)])

        if ibatch == 0:
            centroids = batch_centroids
            counts = batch_counts
        else:
            centroids = np.array([(centroids[i]*counts[i] + batch_centroids[i]*batch_counts[i])/(counts[i] + batch_counts[i]) for i in range(num_clusters)])
            counts = counts + batch_counts

        summaries.append(batch_summaries)

    return summaries, centroids

def kLLMmeans(text_data, 
              num_clusters,
              init = 'k-means++',
              prompt = "", text_type = "",
              force_context_length = 0, max_llm_iter = 5, 
              max_iter = 100, tol=1e-4, random_state = None, 
              emb_type = 'openai', text_features = None,
              final_iter = True,
              initial_iter = True,
              instructor_prompt = "",
              model = "gpt-4o"):
    """
    Runs iterative KMeans clustering with dynamic centroid updates using LLM summaries.
    """
    summaries_evolution = []
    centroids_evolution = []

    if not isinstance(text_data, list):
        raise TypeError(f"Expected a list for variable text_data, but got {type(text_data).__name__}")

    if text_features is None:
        text_features = get_embeddings(text_data, emb_type = emb_type, instructor_prompt = instructor_prompt)
    
    if (final_iter == False) and (initial_iter == False):
        tmp_kmeans_iterations = int(max_iter/(max_llm_iter-1))
    elif (final_iter == False) or (initial_iter == False):
        tmp_kmeans_iterations = int(max_iter/max_llm_iter)
    else: 
        tmp_kmeans_iterations = int(max_iter/(max_llm_iter+1))

    if initial_iter == False:
        kmeans = KMeans(n_clusters=num_clusters, init=init, max_iter=1, random_state=random_state)
    else:
        kmeans = KMeans(n_clusters=num_clusters, init=init, max_iter=tmp_kmeans_iterations, random_state=random_state)
    cluster_assignments = kmeans.fit_predict(text_features)
    centroids = kmeans.cluster_centers_
    
    for iteration in tqdm(range(1, max_llm_iter + 1), desc="Iterating KMeans"):
        # Group texts by cluster
        clustered_texts = {i: [] for i in range(num_clusters)}
        
        # Select texts inside the cluster for summary generation
        if force_context_length > 0:
            clustered_indexes = {i: [] for i in range(num_clusters)}
            for i, cluster in zip(range(len(text_data)), cluster_assignments):
                clustered_indexes[cluster].append(i)
            
            for i_cluster, index_list in clustered_indexes.items():
                cur_cluster_embeddings = [text_features[i] for i in index_list]
                cur_cluster_texts = [text_data[i] for i in index_list] 
                
                if len(index_list)>force_context_length:
                    _, selected_indices = kmeans_plusplus(np.array(cur_cluster_embeddings), n_clusters=force_context_length, random_state=random_state)
                    clustered_texts[i_cluster] = [cur_cluster_texts[i] for i in selected_indices]
                else:
                    clustered_texts[i_cluster] = [cur_text for cur_text in cur_cluster_texts]
        
        else:
            for text, cluster in zip(text_data, cluster_assignments):
                clustered_texts[cluster].append(text)

        # Generate summaries for each cluster
        summaries = [summarize_cluster(clustered_texts[i], prompt, text_type, model) if clustered_texts[i] else "" for i in range(num_clusters)]
        summaries_evolution.append(summaries)

        # Obtain embeddings of summaries
        summary_embeddings = get_embeddings(summaries, emb_type = emb_type, instructor_prompt = instructor_prompt)

        # Check for convergence (if centroid shift is small)
        centroid_shift = np.linalg.norm(centroids - summary_embeddings, axis=1).sum()
        if centroid_shift < tol:
            print(f"Converged after {iteration} iterations.")
            break

        # Update centroids with summary embeddings
        centroids = summary_embeddings
        centroids_evolution.append(centroids)

        if (final_iter == False) and (iteration == max_llm_iter):
            kmeans = KMeans(n_clusters=num_clusters, init=centroids, max_iter=1)
        else:
            kmeans = KMeans(n_clusters=num_clusters, init=centroids, max_iter=tmp_kmeans_iterations)
        
        # Assign data points to the nearest new centroids
        cluster_assignments = kmeans.fit_predict(text_features)

    cluster_centroids = kmeans.cluster_centers_

    return cluster_assignments, summaries, summary_embeddings, cluster_centroids, summaries_evolution, centroids_evolution


def kLLMmedoids(text_data, 
              num_clusters, 
              max_llm_iter = 5, 
              max_iter = 100, tol=1e-4, random_state = None, 
              emb_type = 'openai', labels = [], text_features = None,
              final_iter = False,
              initial_iter = False,
              model = "gpt-4o"):
    """
    Runs iterative KMeans clustering with dynamic centroid updates using LLM summaries.
    """
    
    if not isinstance(text_data, list):
        raise TypeError(f"Expected a list for variable text_data, but got {type(text_data).__name__}")

    if text_features is None:
        text_features = get_embeddings(text_data, emb_type = emb_type)
    
    if (final_iter == False) and (initial_iter == False):
        tmp_kmedoids_iterations = int(max_iter/(max_llm_iter-1))
    elif (final_iter == False) or (initial_iter == False):
        tmp_kmedoids_iterations = int(max_iter/(max_llm_iter+1))
    else: 
        tmp_kmedoids_iterations = int(max_iter/max_llm_iter)
    
    # Step 1: Run KMedioids to initialize centroids
    if initial_iter == False:
        kmedoids = KMedoids(n_clusters=num_clusters, max_iter = 1, random_state=random_state)
    else:
        kmedoids = KMedoids(n_clusters=num_clusters, max_iter = tmp_kmedoids_iterations, random_state=random_state)

    kmedoids.fit(text_features)
    cluster_assignments = kmedoids.labels_
    medoid_indices = kmedoids.medoid_indices_
    medoids = text_features[medoid_indices]
    
    for iteration in tqdm(range(1, max_llm_iter + 1), desc="Iterating KMedoids"):
        # Group texts by cluster
        clustered_texts = {i: [] for i in range(num_clusters)}
        
        # Select texts inside the cluster for summary generation
        for text, cluster in zip(text_data, cluster_assignments):
            clustered_texts[cluster].append(text)

        # Generate summaries for each cluster
        representatives = [representative_cluster(clustered_texts[i], model) if clustered_texts[i] else "" for i in range(num_clusters)]
        
        # Obtain embeddings of summaries
        representative_embeddings = get_embeddings(representatives, emb_type = emb_type)

        # Check for convergence (if centroid shift is small)
        medoids_shift = np.linalg.norm(medoids - representative_embeddings, axis=1).sum()
        if medoids_shift < tol:
            print(f"Converged after {iteration} iterations.")
            break

        # Update centroids with summary embeddings
        medoids = representative_embeddings

        if (final_iter == False) and (iteration == max_llm_iter):
            kmedoids = KMedoids(n_clusters=num_clusters, init=medoids, max_iter=1)
        else:
            kmedoids = KMedoids(n_clusters=num_clusters, init=medoids, max_iter=tmp_kmedoids_iterations)
        
        # Assign data points to the nearest new centroids
        cluster_assignments = kmedoids.fit_predict(text_features)


    return cluster_assignments, representatives