import urllib.parse
import urllib.request
import json

API_KEY = "REPLACE"
CSE_ID = "REPLACE"
k = 10

'''
    https://docs.python.org/3/library/urllib.request.html
    source cited: https://stackoverflow.com/questions/68875116/get-data-from-json-response-from-urllib-using-python
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

if __name__ == "__main__":
    query = input("Please enter your search query: ")
    target_precision = float(input("Enter target precision @10 (decimal from 0-1): "))

    if not (0 <= target_precision <= 1):
        print("Please make sure to enter a correct value. Target precision should be between 0 and 1.")
        exit(1)

    print(f"\Here are the top-10 results for the query: {query}\n")
    results = google_search(query, API_KEY, CSE_ID, k)

    if results:
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Snippet: {result['snippet']}\n")