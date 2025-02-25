import urllib.parse
import urllib.request
import json
import os
import numpy as np
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

load_dotenv()

API_KEY = os.getenv("API_KEY")
CSE_ID = os.getenv("CSE_ID")
k = 10

'''
    debugging source cited:
    https://docs.python.org/3/library/urllib.request.html
    https://stackoverflow.com/questions/68875116/get-data-from-json-response-from-urllib-using-python
'''
def google_search(query, api_key, cse_id, k):
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": cse_id,
        "num": k
    }
    
    url = base_url + "?" + urllib.parse.urlencode(params)
    
    try:
        with urllib.request.urlopen(url) as response:
            results = json.loads(response.read().decode())
    except Exception as e:
        print("Error fetching search results:")
        print(e)
        return []

    if "items" not in results:
        print("No results found.")
        return []

    return [{"title": item.get("title", "no title found"), "url": item.get("link", "no link found"), "snippet": item.get("snippet", "no snippet found")} for item in results["items"]]

def eval(results):
    relevant_docs = []
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Snippet: {result['snippet']}")
        
        while True:
            relevance = input("Is this result relevant? (r/n): ").strip().lower()
            if relevance in ["r", "n"]:
                break
            print("Please enter 'r' or 'n'.")
        
        if relevance == "r":
            relevant_docs.append(result['snippet'])

    precision_at_10 = len(relevant_docs) / len(results) if results else 0

    return(precision_at_10,  relevant_docs)
'''
    source cited: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
'''
def extract_keywords(relevant_docs, num_keywords, current_query_words):
    if not relevant_docs:
        return []
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(relevant_docs)
    feature_names = vectorizer.get_feature_names_out()

    tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)
    sorted_indices = np.argsort(tfidf_scores)[::-1]
    top_keywords = []
    for i in sorted_indices:
        if feature_names[i] not in current_query_words:
            top_keywords.append(feature_names[i])
        if (len(top_keywords) == num_keywords):
            break
    return top_keywords
    
if __name__ == "__main__":
    query = input("Please enter your search query: ")
    target_precision = float(input("Enter target precision @10 (decimal from 0-1): "))

    while not (0 <= target_precision <= 1):
        print("Please make sure to enter a correct value. Target precision should be between 0 and 1.")
        target_precision = float(input("Enter target precision @10 (decimal from 0-1): "))

    while True:
        print(f"Here are the top-10 results for the query: {query}\n")
        results = google_search(query, API_KEY, CSE_ID, k)
        precision_at_10, relevant_docs = eval(results)

        print(f"\nprecision@10 for this iteration: {precision_at_10}")

        if precision_at_10 >= target_precision:
            print(f"Target precision achieved.\n target: {target_precision} \n actual: {precision_at_10}")
            break
        elif precision_at_10 ==0:
            print(f"precision is 0. No relevant results are found. Stopping")
            break
        else:
            print("Target precision not met. Modifying query and re-fetching...")
            current_query_words = query.split(" ")
            keywords = extract_keywords(relevant_docs, 2, current_query_words)
            new_query_list = current_query_words + keywords
            query = (" ").join(new_query_list)
            print(f'new query: {query}')