# CS6111-Project1

# Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisits)
    - [Installation](#installation)
3. [Usage](#usage)
4. [Description of Project](#description-of-project)
    - [Internal Design](#internal-design)
        - [Notable External Libraries Used](#notable-external-libraries-used)
    - [Query-Modification Method](#query-modification-method)

# Introduction
This project is an exploration on query expansion utilizing the Google Custom Search API. With this project, we learned about possible methods of query expansion through word scoring equations such as tf-idf, word placement algorithms using bigrams, and the use of user relevance feedback. It was built for Project 1 of COMS6111 - Advanced Database Systems.

Developed by Matthew Labasan and Phoebe Tang

# Getting Started
## Prerequisites
1. Python 3.8.1 or above
2. Google Custom Search API Key and Google Search Engine Key

## Installation
Clone this repository to your system, navigate to the directory, and run the following lines of code:
1. `python3 -m venv venv`
2. `source ./venv/bin/activate`
3. `pip install -r requirements.txt`

# Usage
1. Run & replace with your parameters, using a query in quotations: 
    - `python main.py <API Key> <Engine Key> <Precision> <Query>`
    - Example usage: `main.py <API Key> <Engine Key> 0.9 “hello world”`
2. Type in Y or N to give user-relevance feedback. Any other letters are not allowed.
3. Sample results can be viwed in the `transcript_nokeys.txt` file.

# Description of Project
## Internal Design
Here, we will outline our `main()` method:
1. Extract the arguments to obtain the search engine ID, API key, precision, and query
2. Enter a loop. The program calls the Google Custom Search API to retrieve search results for a given query. This method is defined as `search()`
    - It uses the build function from googleapiclient.discovery
    - Makes a request to see `cse().list()` with given engine ID
    - Returns a dictionary of the top 10 query results
    - Source Code via [Google](https://github.com/googleapis/google-api-python-client/blob/main/samples/customsearch/main.py)
3. Then, call the `user_relevance()` function to collect user feedback 
    - Iterate through top 10 results and display dictionary content
    - Store relevant and irrelevant documents in separate lists
    - Calculate the precision score and return the score, relevant, and irrelevant lists
4. Next, call our `expand()`and `insert_keywords()` functions to find new keywords and insert them into the new query
    - More details below in query-modification
5. Until target precision is reached, repeat steps 1-4 with the new query
    - Break loop if precision exceeds or equals the target precision
    - Break if we receive a precision of 0 (unable to expand query)

### Notable External Libraries Used
1. `numpy`: For assistance in matrix operations
2. `nltk`: For word tokenization to help find bigrams for keyword insertion
3. `sklearn`: For tf-idf calculations 
4. `googleapiclient`: For tf-idf calculations 

## Query-Modification Method
Our query-expansion algorithm is based on the tf-idf scoring system. The tf-idf score of all words in the relevant documents and the non-relevant documents were separately computed. Then, we removed words from the relevant word list that was present in the non-relevant word list to ensure that only relevant, high scoring tf-idf words were included in our query expansion. 
1. We call the `expand()` function to expand the query
    - It takes in the original query, number of keywords, list of relevant results, and non relevant results
    - We build two corpuses, one containing the words from the title and snippet of relevant documents, and one for the irrelevant documents
    - We then use the tf-idf library from scikit-learn and retrieve the tf-idf scores for the words of each document in both corpuses. We then averaged the scores across the documents of the respective corpus class to get an overall value. 
    - We then prune the top nonrelevant keywords from the top relevant keywords list to make sure that we won’t include any nonrelevant keywords in the new query
    - We then call our insert keywords function to reorder the new query

Our query-word order algorithm is based on the use of bigrams that are present in the relevant document corpus. We worked on two versions of reordering the query – functions insert_keywords and insert_keywords_v2. We ended up using `insert_keywords` but left the `insert_keywords_v2` function for reference
1. `insert_keywords`:

    This method will insert keywords based on present bigrams. Keywords will only be inserted at the start or end of the query since it is assumed that the original user query is in the correct order (we don’t want to risk modification of this order). The logic is below:
    - We get all the bigrams from the document corpus provided (relevant corpus) using the nltk library – methods `word_tokenize()` and `bigrams()`. It is worthy to note that the `word_tokenize()` method does not ignore punctuation.
    - If we have more than 1 keyword:
        - Check for bigrams between each keyword and the start and end query terms. For example, for query “hello world” and keywords (why, there), we would search for the bigrams: (why, hello), (world, why), etc.
        - If a bigram is present, we place the keyword at either the start or end of the existing query, depending on the found bigram.
        - We do this twice, once for each of the two keywords to add.
        - If only one keyword has an associated bigram that allows proper placement, we append the remaining keyword to the end of the query.
        - If no bigrams are found, we check for a bigram of the keywords and use that bigram to determine ordering of the keywords. We then append it at the end of the query.
    - If there is only 1 keyword:
        - Check if it has a bigram with the start or end word of the query
        - Append it in the proper place if so. Else, append it to the end of the query.
2. `insert_keywords_v2`: 
    - We use counter from collections to keep track of the occurrences for bigrams
    - Iterate through all of the documents from the corpus and update the occurrence of each bigram accordingly.
    - We then sort the bigrams by number of occurrences in decreasing order
    - First, we check if the top bigram contains both keywords
        - If so, then we look for any bigram containing (query term, keyword1). If found, then we append the bigram after the query term
    - Next, we iterate through all the query terms.
        - Check if we find any (query term, keyword), if found, insert keyword after the query term and add it to the placed word set
        - Check if we find any (keyword, query term), if found, insert keyword before the query term and add it to the placed word set
    - Finally, we check if there are any keywords left over. If so, just append it to the end of query.
