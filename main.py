import sys
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk import bigrams
from collections import Counter

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

    valid_responses = ['y', 'n']
    while True:
      if answer.lower() in valid_responses:
        if answer.lower() == 'y':
          precision += 1
          relevant_results.append(pruned_result)
        else:
          nonrelevant_results.append(pruned_result)
        break
      answer = input("Please make sure to input Y or N. Relevant (Y/N)? ")
  
  # in case we returned less than 10 docs or 0 docs
  precision = precision/len(results) if len(results) > 0 else 0
  return precision, relevant_results, nonrelevant_results

def insert_keywords_v2(query, keywords, corpus):
    """
    Optimally order new keywords into query by searching for the most frequent bigrams present in the corpus.
    
    Args:
        query (str): Original query
        keywords (list): Words to append & order into query
        corpus (list): List of documents (strings)
    
    Returns:
        new_query (str): Updated query
    """
    new_query = query
    query_words = new_query.split(" ")
    bigram_counts = Counter()
    
    # Get bigrams from corpus and count occurrences
    for document in corpus:
        tokens = word_tokenize(document)
        doc_bigrams = list(bigrams(tokens))
        for bigram in doc_bigrams:
            if any(word in bigram for word in query_words + keywords):
                bigram_counts[bigram] += 1
    
    # Sort bigrams by count
    sorted_bigrams = sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Determine best order for keywords

    top1, top2 = sorted_bigrams[0]
    placed_keywords = set()

    # if top bigram is 2 keywords
    if top1 in keywords and top2 in keywords:
       # append them after a query word if the bigram exists
       for q in query_words:
          if (q, top1) in bigram_counts:
              query_words.insert(query_words.index(q) + 1, top1)
              query_words.insert(query_words.index(q) + 2, top2)
              placed_keywords.update([top1, top2])
              break
    else:
      for (word1, word2), _ in sorted_bigrams:
          if word1 in query_words and word2 in keywords and word2 not in placed_keywords:
              query_words.insert(query_words.index(word1) + 1, word2)
              placed_keywords.add(word2)
          elif word2 in query_words and word1 in keywords and word1 not in placed_keywords:
              query_words.insert(query_words.index(word2), word1)
              placed_keywords.add(word1)
    
    for keyword in keywords:
        if keyword not in placed_keywords:
            query_words.append(keyword)
    
    return " ".join(query_words)

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
  
  # Status of bigram search for each keyword & final check
  bigram1_found = False
  bigram2_found = False
  keyword_bigram_found = False
  
  # Get bigrams from corpus
  all_bigrams = []
  for document in corpus:
    # Word_tokenize considers punctuation, removing non-word bigrams
    tokens = word_tokenize(document)
    all_bigrams += list(bigrams(tokens))
  
   # Check if a keyword and start/end query term is a bigram. Assumes original query has good ordering, so no middle placements!
  if len(keywords) > 1:
    # KEYWORD #1
    query_words = new_query.split(" ")
    for bigram in all_bigrams: # Exits loop once ONE keyword is assigned. If none is found, it'll check again (redundant for now).
      if bigram1_found:
        break
      # Go through keyword and query start/end combinations. 
      for keyword in keywords:
        if not bigram1_found and (keyword == bigram[0] and query_words[0] == bigram[1]): # Bigram = (keyword1, starting query word) -> Add to start
          new_query = keyword + " " + new_query
          bigram1_found = True
          break
        elif not bigram1_found and (keyword == bigram[1] and query_words[-1] in bigram[0]): # Bigram = (ending query word, keyword1) -> Add to end
          new_query = new_query + " " + keyword
          bigram1_found = True
          break
    
    # KEYWORD #2
    query_words = new_query.split(" ") # Redo incase query was updated
    for bigram in all_bigrams:
      if bigram2_found:
        break
      # Go through keyword and query start/end combinations
      for keyword in keywords:
        if not bigram2_found and (keyword == bigram[0] and query_words[0] == bigram[1]): # Bigram = (keyword2, starting query word) -> Add to start
          new_query = keyword + " " + new_query
          bigram2_found = True
          break
        elif not bigram2_found and (keyword == bigram[1] and query_words[-1] in bigram[0]): # Bigram = (ending query word, keyword1) -> Add to end
          new_query = new_query + " " + keyword
          bigram2_found = True
          break
  
    # CHECK: If one keyword was not added, Append to end
    if not bigram1_found and bigram2_found:
      bigram1_found = True # Ensure no more checks go through
      new_query = new_query + " " + keywords[0]
    if not bigram2_found and bigram1_found:
      new_query = new_query + " " + keywords[1]
      bigram2_found = True
      
    # CHECK: If two keywords not added, check if bigram, reorder the two if needed, then append to end
    if not bigram1_found and not bigram2_found: 
      for bigram in all_bigrams:
        if keywords[0] in bigram and keywords[1] in bigram:
          keywords = list(bigram)
          new_query = new_query + " " + keywords[0] + " " + keywords[1] 
          keyword_bigram_found = True
        break
      
      if not keyword_bigram_found:
        new_query = new_query + " " + keywords[0] + " " + keywords[1] # No special ordering
      
  else:
    # If only one keyword present
    query_words = new_query.split(" ")
    for bigram in all_bigrams:
      if not bigram1_found and (keyword == bigram[0] and query_words[0] == bigram[1]): # (keyword1, starting query word)
        new_query = keyword + " " + new_query
        break
      elif not bigram1_found and (keyword == bigram[1] and query_words[-1] in bigram[0]): # (ending query word, keyword1)
        new_query = new_query + " " + keyword
        break
    
    # CHECK: If keyword was not added, Append to end
    if not bigram1_found:
      new_query = new_query + " " + keywords[0]

  # Idea of this: search if bigrams including the start and end query terms with a keyword. If yes, append to that side. Do not append to middle
  # as we assume it is correct and wanted by the user. If no bigrams found, try to bigram keywords. Else, append to end.

  # NOTES
  # Search for bigrams -- we could do a nltk.bigram finder in our corpus, however:
  # If our keywords were, (New, York), and a bigram result is (York, New) from something like "...York. New...", it would choose this! But wrong.
  # So, if an earlier, subpar bigram shows up earlier in the all_bigrams list, won't be optimal. Not sure how to fix?
  # Actually, punctuation will now make (York New), it'd be (York, .)
  # I was also thinking of doing a thing where if only one word is expanded, then use a bigram to derive a second word, but that 
  # defeats the whole purpose of removing it if a word is present in nonrelev.
  # Note: need seperate checks for keyword 1 and 2 to ensure proper indexing when using reorder function. kinda messy but whatever
  # - See expand() bottom for one possible removal
  
  return new_query

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
  current_query_words = query.lower().split(" ") # Standardize word
  
  print("Indexing results ....")
  
  # Build document corpus' and 
  for result in relevant_results:
    relevant_corpus.append(result["title"] + " " + result["snippet"]) # Format: "Title Snippet"
  for result in nonrelevant_results:
    nonrelevant_corpus.append(result["title"] + " " + result["snippet"]) 

  if relevant_results: 
    # Obtain Tfidf scores averaged across relevant documents
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(relevant_corpus) 
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)
    sorted_indices = np.argsort(tfidf_scores)[::-1] # Indexes of feature_names
    
    for i in sorted_indices:
        feature_name_lower = feature_names[i].lower() # Standardize
        if feature_name_lower not in current_query_words:
            top_relevant_keywords.append(feature_name_lower)
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
        feature_name_lower = feature_names[i].lower() # Standardize
        if feature_name_lower not in current_query_words:
            top_nonrelevant_keywords.append(feature_name_lower)
        if (len(top_nonrelevant_keywords) == num_keywords):
            break
          
  # Prune top_nonrelevant_keywords from top_relevant_keywords list. Ensures query won't include nonrelevant terms.
  for keyword in top_relevant_keywords:
    if keyword in top_nonrelevant_keywords:
      top_relevant_keywords.remove(keyword)
  
  # Add keywords to query
  if len(top_relevant_keywords) > 1:
    print("Augmenting by " + top_relevant_keywords[0] + " " + top_relevant_keywords[1])
    new_query = insert_keywords_v2(query, top_relevant_keywords[0:2], relevant_corpus)
  elif len(top_relevant_keywords) == 1:
    print("Augmenting by " + top_relevant_keywords[0])
    new_query = insert_keywords_v2(query, top_relevant_keywords, relevant_corpus) # Only one elem in array, so no indexing
  else:
    # If no top_relevant_results remaining after pruning, use two lowest scoring nonrelevant document words. THOUGHTS? 
    # (I don't think this is possible if relevancy is marked correctly, but...) 
    print("Augmenting by " + top_nonrelevant_keywords[-2] + " " + top_nonrelevant_keywords[-1])
    new_query = insert_keywords_v2(query, top_nonrelevant_keywords[-2:], relevant_corpus) # Use relevant corpus for bigram!
  
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
        print("======================")
        print("FEEDBACK SUMMARY")
        print(f"Query: {query}")
        print(f"Precision: {result_precision}")
        print(f"Still below the desired precision of {goal_precision}")
        query = expand(query, 2, relevant_results, nonrelevant_results)
        
        # Check Validity
        if not query:
          print("Below desired precision, but can no longer augment the query")
          break
    else: 
      break # Error fetching results
  
  print("======================")
  print("FEEDBACK SUMMARY")
  print(f"Query: {query}")
  print(f"Precision: {result_precision}")
  print("Desired precision reached, done")
  
if __name__ == "__main__":
  main()