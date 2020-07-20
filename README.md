# Indeed.com-jobs-Classifier
Objective: Scrapping various jobs on Indeed.com and map them into predefined categories based on job title, summary and other relevant features.

Modularized the problem into 2 sub-problems:
1. Scrapper module
2. Mapper module

## Scrapper Module
Utilised packages like BeautifulSoup, urllib, pandas, numpy. Scrapped as many jobs as possible on Indeed.com based on company names in an input csv file. Scrapeed the following details for each role:
                       1. Job title
                       2. Company
                       3. Location
                       4. Summary
                      
The scrapping process invovled parsing the URL and while the status_code is 200 for each webpage the parser created, then the jobs on that webpage is extracted through BeautifulSoup librabry calls. Once you have the required details from the webpage, then it is written to a intermediate pandas dataframe and finally converted to a csv file as a columnar format.

Important note - If you do not add a pause factor while moving fro one webpage to another using the time.sleep() function then the website's admin would block your IP address for a certain period.

## Mapper Module
The requirement is to map the extracted jobs into 6 categories like:
1. Sales and Marketing
2. Engineering, Research and Development
3. Customer Services
4. Business Operations
5. Leadership
6. Other

One of the constraints in this problem is that we do not have labeled data to make it a typical supervised learning. The method used in this project focuses on building a classifier that has the right feature set and the right optimization stargtegy in the deciding the feature weights.

The classifier relies on processed text input and calculates 2 major factors - lexical similarity and semantic similarity.
## Lexical Similarity
This measures word-wise similarity and gives the similarity score based
individual word distances in the embedded dimensions. This is done by the Spacy’s Model
and this can be used to handle the Job Title feature.
## Semantic Similarity 
This takes multiple words into account to provide a similarity score
based on semantics. This is handled by the Glove model that processes the Job summary
feature in this problem.

For further intricate technical details, kindly do refer the project report file.
