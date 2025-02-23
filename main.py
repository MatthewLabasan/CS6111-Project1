# from dotenv import load_dotenv
import os
import sys
import json
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer

# use scikit learn for tf-idf calculations

# load_dotenv()
# GSEID = os.getenv("GSEID")
# GSAPI = os.getenv("GSAPI") 

def search(GSAPI, GSEID, query) -> dict:
  service = build(
        "customsearch", "v1", developerKey=GSAPI
    )
  res = (
      service.cse()
      .list(
          q=query,
          cx=GSEID,
      )
      .execute()
  )
  
  return res


# Check Relevance for 10 Documents, returning nonrelevant and relevant document information
def user_relevance(results):
  precision = 0
  relevant_results = []
  nonrelevant_results = []
  
  for index, result in enumerate(results):
    pruned_result = {"URL": result['link'], "Title": result['title'], "Summary": result['snippet']}
    
    # Format Print
    print(f"Result {index+1}")
    print("[")
    for key, value in pruned_result.items():
      print(f" {key}: {value}")
    print("]")
    
    answer = input("Relevant (Y/N)? ")
    
    if answer == "y" or answer == "Y":
      precision += 1
      relevant_results.append(pruned_result)
    else:
      nonrelevant_results.append(pruned_result)
  
  precision = precision/10
  return precision, relevant_results, nonrelevant_results

def expand(relevant_results, nonrelevant_results, query):
  # Status
  print("Indexing results ....")
  
  # Build Summary Corpus
  relevant_summaries = []
  for result in relevant_results:
    relevant_summaries.append(result["Summary"])
    
  # Import Stop_Words
  stop_words = None
  try:
    with open("./stopwords.json", "r") as file:
      stop_words = list(json.load(file)["stop_words"])
      file.close()
  except Exception as e:
      "Error: Unable to import stop words."
  
  # Get TF-IDF Scores (Highest over all documents)
  vectorizer = TfidfVectorizer(stop_words=stop_words)
  score_array = vectorizer.fit_transform(relevant_summaries) # Returns sparse matrix (position (row, column), score)
  names = vectorizer.get_feature_names_out()
  
  # -------
  # APPROACH 1: Highest TF-IDF Over All Documents (weird because term frequency varies and we are just taking the max score, so focuses more on words with very relevant context, but probably more general words rather than niche to the desired query)
  # There seems to be a problem with the stop words its not properly removing bc of normalization or something. Problem bc they have high TF scores, so they are the best candidates in this approach.
  # -------
  # Get TF-IDF Scores (Highest over all documents)
  vectorizer = TfidfVectorizer()
  score_array = vectorizer.fit_transform(relevant_summaries) # Returns sparse matrix (position (row, column), score)
  names = vectorizer.get_feature_names_out()
  
  score_array = score_array.toarray()
  max_scores = score_array.max(axis=0) # Get max TF-IDF score among all the documents (most frequent (in a certain doc) and most rare)
  
  print(max_scores)
  print(names)
  
  # -------
  # APPROACH 2: Highest IDF Over All Documents (level playing, no term frequency -> more specific results (that may be skewed, but faster?))
  # Use Tfidf with just idf values, then take the max words from the sparse matrix
  # -------
  
  # Find Top 2 Scoring Words (ensure not query words)
    

  # Indexing results .... (nonrelevant? / placement?)
  
  print(f"Augmenting by ...")

def main():
  # Get arguments
  try:
    GSAPI = sys.argv[1]
    GSEID = sys.argv[2]
    goal_precision = int(sys.argv[3])
    query = sys.argv[4]
  except Exception as e:
    print("Usage: python main.py <API Key> <Engine Key> <Precision> <Query>")

  # Relevance Loop
  result_precision = -1
  while result_precision < goal_precision:
    # Google Search
    print(f"""Parameters:
    {'Client key:':<12} = {GSAPI}
    {'Engine key:':<12} = {GSEID}
    {'Query:':<12} = {query}
    {'Precision:':<12} = {goal_precision}
    """)
    print("Google Search Results:\n======================")
    results = search(GSAPI, GSEID, query)['items']
    
    # Check Relevance
    result_precision, relevant_results, nonrelevant_results = user_relevance(results)
    
    if result_precision < goal_precision:
      print(f"Precision: {result_precision}")
      print(f"Still below the desired precision of {goal_precision}")
      query = expand(relevant_results, nonrelevant_results, query)
  
if __name__ == "__main__":
  main()