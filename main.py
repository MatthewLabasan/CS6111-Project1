import sys
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk import bigrams

nltk.download('punkt_tab')

def search(GSAPI, GSEID, query) -> dict:
  """
  Call Google API.
  Source: https://github.com/googleapis/google-api-python-client/blob/main/samples/customsearch/main.py
  
  Args:
      GSAPI (str): Google Search API Key
      GSEID (str): Google Search Engine ID
      query (str): Desired query

  Returns:
      res (dict): Dictionary of 10 results
  """
  
  service = build(
        "customsearch", "v1", developerKey=GSAPI
    )
  
  try:
    res = (
        service.cse()
        .list(
            q=query,
            cx=GSEID,
        )
        .execute()
    )
  except Exception as e:
        print("Error fetching search results:")
        print(e)
        return []
  
  return res


def user_relevance(results):
  """
  Check Relevance for 10 Documents, returning nonrelevant and relevant document information
  
  Args:
      results (dict): Dictionary of n results

  Returns:
      precision (float): New precision value 
      relevant_results (list): Relevant documents. 
      nonrelevant_results(list): Nonrelevant documents
        - Each item in both lists is a dict containing keys "url", "title", "summary"
  """
  precision = 0
  relevant_results = []
  nonrelevant_results = []
  
  for index, result in enumerate(results):
    pruned_result = {"url": result['link'], "title": result['title'], "snippet": result['snippet']}
    
    # Format Print
    print(f"Result {index+1}")
    print("[")
    for key, value in pruned_result.items():
      print(f" {key}: {value}")
    print("]\n")
    
    answer = input("Relevant (Y/N)? ")
    
    if answer == "y" or answer == "Y":
      precision += 1
      relevant_results.append(pruned_result)
    else:
      nonrelevant_results.append(pruned_result)
  
  precision = precision/10
  return precision, relevant_results, nonrelevant_results

def reorder(query_words, index, keyword, bigram):
  """
  Reorders keyword into the query if a bigram between a query word and a keyword is present.
  Does not reorder if a pair of query_words being seperat
  
  Args:
      query_words (list): Query word tokens
      index (int):  Index of query word in bigram
      keyword (str): Keyword to add
      bigram (tuple): Specific ordering of words

  Returns:
      new_query (str): Updated query
  """
  new_query = ""
  for query_index, query_word in query_words:
    if query_index == index:
      new_query += keyword
    else:
      new_query += query_word
      
  return new_query

def insert_keywords(query, keywords, corpus):
  """
  Optimally order new keywords into query by searching for a bigram present in corpus.
  Checks if exists a bigram for (query word & keyword1), (query word & keyword2), (keyword1 & keyword2)
  
  Args:
      query (str): Original query
      keywords (list):  Words to append & order into query
      corpus (list): List of documents (strings)

  Returns:
      new_query (str): Updated query
  """
  # This variable may change more than once throughout the function
  new_query = query
  
  # Status of search for each keyword
  bigram1_found = False
  bigram2_found = False
  
  # Get bigrams from corpus
  all_bigrams = []
  for document in corpus:
    # Note: word_tokenize considers punctuation, which is fine as this will help remove unrealistic bigrams
    # Ex. "Pepperoni pizza, coffee" (pizza, ,) vs (pizza, coffee)
    tokens = word_tokenize(document)
    all_bigrams += list(bigrams(tokens))
  
   # Check if a keyword and query term is a bigram
  if len(keywords) > 1:
    # Keyword #1
    query_words = new_query.split(" ")
    for bigram in all_bigrams:
      if bigram1_found:
        break
      # Go through all query and keyword combinations
      for index, query_word in enumerate(query_words):
        if not bigram1_found and (keywords[0] in bigram and query_word in bigram):
          new_query = reorder(query_words, index, keywords[0], bigram)
          bigram1_found = True
          break
    
    # Keyword #2
    query_words = new_query.split(" ") # Redo incase query was updated
    for bigram in all_bigrams:
      if bigram2_found:
        break
      # Go through all query and keyword combinations
      for index, query_word in enumerate(query_words):
        if not bigram2_found and keywords[1] in bigram and query_word in bigram:
          new_query = reorder(query_words, index, keywords[1], bigram)
          bigram2_found = True
          break
      
    # Check if two keywords is a bigram (switch to more optimal order if not added in already)
    if not bigram1_found and not bigram2_found: 
      for bigram in all_bigrams:
        if keywords[0] in bigram and keywords[1] in bigram:
          keywords = list(bigram)
          new_query = new_query + " " + keywords[0] + + " " + keywords[1] 
        break
  else:
    # If only one keyword present
    for bigram in all_bigrams:
      if bigram1_found == True:
        break
      # Go through all query and keyword combinations
      for index, query_word in enumerate(query_words):
        if not bigram1_found and (keywords[0] in bigram and query_word in bigram):
          new_query = reorder(query_words, index, keywords[0], bigram)
          bigram1_found = True
          break

  
  # Search for bigrams -- we could do a nltk.bigram finder in our corpus, however:
  # If our keywords were, (New, York), and a bigram result is (York, New) from something like "...York. New...", it would choose this! But wrong.
  # So, if an earlier, subpar bigram shows up earlier in the all_bigrams list, won't be optimal. Not sure how to fix?
  # Actually, punctuation will now make (York New), it'd be (York, .)
  # I was also thinking of doing a thing where if only one word is expanded, then use a bigram to derive a second word, but that 
  # defeats the whole purpose of removing it if a word is present in nonrelev.
  # Note: need seperate checks for keyword 1 and 2 to ensure proper indexing when using reorder function. kinda messy but whatever
  
  query = query + ", KEYWORDS: " + keywords[0] + ", " + keywords[1]
  print(query)


def expand(query, num_keywords, relevant_results, nonrelevant_results):
  """
  Expand query with two new words based on user relevance results.
  Tfidf section by Phoebe Tang (adironene)
  
  Args:
      query (str): Original query
      num_keywords (int): Number of keywords to create 
      relevant_results (list): Relevant documents
      nonrelevant_results(list): Nonrelevant documents

  Returns:
      new_query (str): new query, else None if no relevant results
  """
  relevant_corpus = []
  nonrelevant_corpus = []
  top_relevant_keywords = []
  top_nonrelevant_keywords = []
  expansion_keywords = []
  current_query_words = query.split(" ")
  
  print("Indexing results ....")
  
  # Build document corpus' and 
  for result in relevant_results:
    relevant_corpus.append(result["title"] + ", " + result["snippet"]) # Format: "Title, Snippet"
  for result in nonrelevant_results:
    nonrelevant_corpus.append(result["title"] + ", " + result["snippet"]) 

  if relevant_results: 
    # Obtain Tfidf scores averaged across relevant documents
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(relevant_corpus) 
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)
    sorted_indices = np.argsort(tfidf_scores)[::-1] # Indexes of feature_names
    
    for i in sorted_indices:
        if feature_names[i] not in current_query_words:
            top_relevant_keywords.append(feature_names[i])
        if (len(top_relevant_keywords) == num_keywords):
            break
  else:
    # If no relevant results, end
    return None
          
  if nonrelevant_results: 
    # Obtain Tfidf scores averaged across nonrelevant documents
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(nonrelevant_corpus)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)
    sorted_indices = np.argsort(tfidf_scores)[::-1] # Indexes of feature_names
    
    for i in sorted_indices:
        if feature_names[i] not in current_query_words:
            top_nonrelevant_keywords.append(feature_names[i])
        if (len(top_nonrelevant_keywords) == num_keywords):
            break
          
  # Prune top_nonrelevant_keywords from top_relevant_keywords list. Ensures query won't include nonrelevant terms.
  for keyword in top_relevant_keywords:
    if keyword in top_nonrelevant_keywords:
      top_relevant_keywords.remove(keyword)
  
  # Add keywords to query
  if len(top_relevant_keywords) > 1:
    print("Augmenting by " + top_relevant_keywords[0] + " " + top_relevant_keywords[1])
    new_query = insert_keywords(query, top_relevant_keywords[0:2], relevant_corpus)
  elif len(top_relevant_keywords) == 1:
    print("Augmenting by " + top_relevant_keywords[0])
    new_query = insert_keywords(query, top_relevant_keywords[0], relevant_corpus)
  else:
    # If no top_relevant_results remaining after pruning, use two lowest scoring nonrelevant document words. THOUGHTS? 
    # (I don't think this is possible if relevancy is marked correctly, but...) 
    print("Augmenting by " + top_nonrelevant_keywords[-2] + " " + top_nonrelevant_keywords[-1])
    new_query = insert_keywords(query, top_nonrelevant_keywords[-2:], relevant_corpus) # Use relevant corpus for bigram!
  
  return new_query


def main():
  # Get arguments
  try:
    GSAPI = sys.argv[1]
    GSEID = sys.argv[2]
    goal_precision = float(sys.argv[3])
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
    if results != None: 
      # Check relevance
      result_precision, relevant_results, nonrelevant_results = user_relevance(results)
      if result_precision < goal_precision:
        print(f"Precision: {result_precision}")
        print(f"Still below the desired precision of {goal_precision}")
        query = expand(query, 2, relevant_results, nonrelevant_results)
        
        # Check Validity
        if not query:
          print("Below desired precision, but can no longer augment the query")
          break
    else: 
      break # Error fetching results
  
if __name__ == "__main__":
  main()