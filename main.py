# from dotenv import load_dotenv
import os
import sys
import json
from googleapiclient.discovery import build
import 

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


# Check Relevance for 10 Documents and Print Results as you go
def user_relevance(results):
  relevance = 0
  
  for index, result in enumerate(results):
    print(f"[\nResult {index+1}\n")
    print(f"URL: {result['link']}")
    print(f"Summary: {result['snippet']}\n]\n")
    answer = input("Relevant (Y/N)? ")
    
    if answer == "y" or answer == "Y":
      relevance += 1
  
  relevance = relevance/10
  return relevance

def expand(results):
  x=10
  # Indexing results ....
  # Indexing results ....
  # Augmenting by  spiral galaxy

def main():
  # Get arguments
  try:
    GSAPI = sys.argv[1]
    GSEID = sys.argv[2]
    precision = int(sys.argv[3])
    query = sys.argv[4]
  except Exception as e:
    print("Usage: python main.py <API Key> <Engine Key> <Precision> <Query>")

  # Relevance Loop
  result_precision = -1
  while result_precision < precision:
    # Google Search
    print(f"""Parameters:
    {'Client key:':<12} = {GSAPI}
    {'Engine key:':<12} = {GSEID}
    {'Query:':<12} = {query}
    {'Precision:':<12} = {precision}
    """)
    print("Google Search Results:\n======================")
    results = search(GSAPI, GSEID, query)['items']
    
    # Check Relevance
    result_precision = user_relevance(results)
    
    if result_precision < precision:
      print(f"Precision: {result_precision}")
      print(f"Still below the desired precision of {precision}")
    else:
      break
    
    # Update Query
    # query = expand(results, query)
  
  
  
  

if __name__ == "__main__":
  main()