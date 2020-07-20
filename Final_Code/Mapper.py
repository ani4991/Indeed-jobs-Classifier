import requests
import bs4
from bs4 import BeautifulSoup
import re, urllib
from urllib.request import urlopen
import pandas as pd
import matplotlib as plt
import csv, time, os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import numpy as np
from collections import Counter
from glob import glob
import collections
 
# Text Pre-processing declarations
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


## Loading the Pre-trained Spacy word Model

import spacy,en_core_web_lg
nlp = en_core_web_lg.load()

## Loading the Universal encoder Glove model

import numpy as np
def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile, encoding='utf8')
    model = {}
    for line in f:
        values = line.split()
        word = ''.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        model[word] = coefs
    f.close()
    print ("Done.",len(model)," words loaded!")
    return model


import re
from nltk.corpus import stopwords
import pandas as pd

def preprocess(raw_text):

    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split 
    words = letters_only_text.lower().split()

    # remove stopwords
    stopword_set = set(stopwords.words("english"))
    cleaned_words = list(set([w for w in words if w not in stopword_set]))

    return cleaned_words

## Calculating the cosine diatnce using Scipy's spatial distance metric

def cosine_distance_between_two_words(word1, word2):
    import scipy
    return (1- scipy.spatial.distance.cosine(model[word1], model[word2]))


gloveFile = "C:\\Users\\ani49\\Data_Science_Case_Studies\\Enlyft\\glove.840B.300d.txt"
glove_model = loadGloveModel(gloveFile)

## Calculating the semantic similarity using the Glove Model

def cosine_distance_wordembedding_method(s1, s2):
    import scipy
    vector_1 = np.mean([glove_model[word] for word in preprocess(s1)],axis=0)
    vector_2 = np.mean([glove_model[word] for word in preprocess(s2)],axis=0)
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    #print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')
    return 1-cosine

## Cleaning the text sentences so that punctuation marks, stop words &amp; digits are removed

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    processed = re.sub(r"\d+","",normalized)
    y = processed.split()
    return y

## Cleaning and formatiing the two feature columns - "Job_titles" and "Job_summary"

def clean_features(input_data,input_data_job_summary):
  
    input_cleaned_data=[]
    input_cleaned_data_job_summary = []

    #print("input_data: ", len(input_data))
    #print("input_data_job_summary:", len(input_data_job_summary))
    
    for i,j in zip(input_data,input_data_job_summary):

        # job title
        line = i.strip()
        cleaned = clean(line)
        cleaned = ' '.join(cleaned)
        input_cleaned_data.append(cleaned)

        # job summary
        line = j.strip()
        cleaned = clean(line)
        cleaned = ' '.join(cleaned)
        input_cleaned_data_job_summary.append(cleaned)
    
    return input_cleaned_data, input_cleaned_data_job_summary

## Calculate the model scores based on selected feature weight types

def calc_score(f_w_t, title_scores, summary_scores):
    
    if f_w_t == '0,1':
        return summary_scores
    elif f_w_t == '1,0':
        return title_scores
    elif f_w_t == '1,1':
        return (title_scores + summary_scores) / 2
    elif f_w_t == '1.5,1':
        return ((1.5 * title_scores) + summary_scores) / 2.5
    elif f_w_t == '1,1.5':
        return (title_scores + (1.5 * summary_scores )) / 2.5
    elif f_w_t == '1,2':
        return (title_scores + (2 * summary_scores)) / 3
    elif f_w_t == '2,1':
        return ((2 * title_scores) + summary_scores) / 3
    elif f_w_t == '2,1.5':
        return ((2 * title_scores) + (1.5 * summary_scores)) / 3.5
    elif f_w_t == '1.5,2':
        return ((1.5 * title_scores) + (2 * summary_scores)) / 3.5
    elif f_w_t == '2,2':
        return ((2 * title_scores) + (2 * summary_scores)) / 4
    elif f_w_t == '3,1':
        return ((3 * title_scores) + summary_scores) / 4
    elif f_w_t == '1,3':
        return (title_scores + (3 * summary_scores)) / 4

## Calculating the Metric - Intra-Cluster tightness of each cluster for list of feature weights

def calc_intra_cluster_tightness(categorized_job_titles, categorized_job_summary):
    
    title_scores = []
    summary_scores = []
    
    for key in categorized_job_titles:
        title_list = categorized_job_titles[key]
        score=0
        for i in range(len(title_list)):
            for j in range(i+1, len(title_list)):
                score += title_list[i].similarity(title_list[j])
        
        title_scores.append(score)
        
    for key in categorized_job_summary:
        summary_list = categorized_job_summary[key]
        score=0
        for i in range(len(summary_list)):
            for j in range(i+1, len(summary_list)):
                score += summary_list[i].similarity(summary_list[j])
        
        summary_scores.append(score)
        
    return ((sum(title_scores) / 5) + (sum(summary_scores) / 5)) / 2

## Initial mapping of all the jobs is done to optimize the feature weights pair that can be used in the Actual_Mapper function
 
def optimization_mapper(f_w_t, input_cleaned_data,input_cleaned_data_job_summary,output_cleaned_data):
    categorized_job_titles = collections.defaultdict(list)
    categorized_job_summary = collections.defaultdict(list)
    
    for title,summary in zip(input_cleaned_data,input_cleaned_data_job_summary):
        token_title = nlp(title)
        token_summary = nlp(summary)
        feature_score = {}
        ctr=0
        
        for category in output_cleaned_data:

            token_category = nlp(category)
            title_scores = token_title.similarity(token_category)
            if title_scores < 0.50:
                ctr += 1
            summary_scores = token_summary.similarity(token_category)
            #summary_scores = cosine_distance_wordembedding_method(summary, category)
            feature_score[category] = calc_score(f_w_t, title_scores, summary_scores)
        
        if ctr == len(output_cleaned_data):
            categorized_job_titles['other'].append(token_title)
            categorized_job_summary['other'].append(token_summary)


        else:
            maximum = max(feature_score, key=feature_score.get)
            categorized_job_titles[maximum].append(token_title)
            categorized_job_summary[maximum].append(token_summary)
            
    return calc_intra_cluster_tightness(categorized_job_titles, categorized_job_summary)

## Mapping the jobs to respective categories after finding the optimum feature weights

def Actual_mapper(best_f_w_t, input_cleaned_data,input_cleaned_data_job_summary,output_cleaned_data):
    

    score_results = {}
    final_results = {'other': 0,
                     'engineering research development': 0,
                     'customer service': 0,
                     'business operation': 0,
                     'sale marketing':0,
                     'leadership':0}

    job_id=0
    for title,summary in zip(best_f_w_t, input_cleaned_data,input_cleaned_data_job_summary):
        feature_score = {}
        ctr=0

        token_title = nlp(title)
        token_summary = nlp(summary)

        for category in output_cleaned_data:

            token_category = nlp(category)
            title_scores = token_title.similarity(token_category)
            if title_scores < 0.50:
                ctr += 1
            summary_scores = token_summary.similarity(token_category)
            #summary_scores = cosine_distance_wordembedding_method(summary, category)
            feature_score[category] = calc_score(best_f_w_t, title_scores, summary_scores)

        if ctr == len(output_cleaned_data):
            score_results[job_id] = 'other'
            if 'other' in final_results:
                final_results['other'] += 1
            else: 
                final_results['other'] = 1

        else:
            maximum = max(feature_score, key=feature_score.get)
            score_results[job_id] = maximum
            if maximum in final_results:
                final_results[maximum] += 1
            else: 
                final_results[maximum] = 1

        job_id += 1
    return final_results

## Appending the classifications to a result dataframe

def appending_results(company,final_count, final_results):
    

    final_count = final_count.append({'Company':company, 'Number of Jobs': sum(final_results.values()), 
                                     'sale marketing':final_results['sale marketing'], 
                                      'engineering research development':final_results['engineering research development'],
                                      'customer service': final_results['customer service'],
                                     'business operation':final_results['business operation'], 
                                      'leadership':final_results['leadership'],
                                     'other':final_results['other']}, ignore_index=True)
    #print(final_count)
    return final_count

if __name__ == "__main__":

    

    companies_list_dict = {}
    filenames = glob('C:\\Users\\ani49\\Data_Science_Case_Studies\\Enlyft\\dataset_companies\\*.csv')

    li = []

    for f in filenames:
        df = pd.read_csv(f, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)



    for f in filenames:
        company_name = os.path.basename(f).split('.')
        companies_list_dict[company_name[0]] = pd.read_csv(f)
        
    for key,value in companies_list_dict.items():
        print(key,value.shape)

        
    final_count = pd.DataFrame(columns=['Company','Number of Jobs','sale marketing',
    'engineering research development',
    'customer service',
    'business operation',
    'leadership','other'])


    job_categories= ['Sales & Marketing', 'Engineering, Research & Development', 'Customer Services', 'Business Operations', 'Leadership']
    output_cleaned_data = []
    for j in job_categories:
        line = j.strip()
        cleaned = clean(line)
        cleaned = ' '.join(cleaned)
        output_cleaned_data.append(cleaned)

    ## optimization code
    """Here we try a list of feature weights and calculate the weighted 
    average of both job_titles and job_summary, look for the pair of weights 
    that has the maximum intra-cluster tightness i.e the categorized jobs are 
    similaer to each other implying they must belong to the same cluster."""

    # using the large data of all the companies to get the optimal feature weights
    # DataFrame used - 'frame'

    frame = frame.dropna()
    input_data,input_data_job_summary = frame['Job_title'].tolist(), frame['Summary'].tolist()
    input_cleaned_data, input_cleaned_data_job_summary = clean_features(input_data,input_data_job_summary)
    feature_weights_type = ['0,1','1,0','1,1','1,2','2,1','2,2','3,1' ,'1,3']
    feature_weights_scores = {}

    for f_w_t in feature_weights_type:
        feature_weights_scores[f_w_t] = optimization_mapper(f_w_t, input_cleaned_data,input_cleaned_data_job_summary,output_cleaned_data)

    # Extracting the best feature weights from the feature_weights_scores dictionary
    best_f_w_t = max(feature_weights_scores, key=feature_weights_scores.get)


    ## Categorization code

    """Once we have found the optimal feature weights, 
    we can now use them to categorize the incoming job 
    titles for each company into the given 6 categories"""

    for key,value in companies_list_dict.items():
        df = value
        df=df.dropna()
        print(key,df.shape)
        input_data,input_data_job_summary = df['Job_title'].tolist(), df['Summary'].tolist()
        #print(len(input_data), len(input_data_job_summary))
        input_cleaned_data, input_cleaned_data_job_summary = clean_features(input_data,input_data_job_summary)
        count_values_dict = Actual_mapper(best_f_w_t, input_cleaned_data,input_cleaned_data_job_summary,output_cleaned_data)
        final_count = appending_results(key,final_count, count_values_dict)

    ## Saving the DataFrame to an "Output.csv" file

    final_count.to_csv('C:\\Users\\ani49\\Data_Science_Case_Studies\\Enlyft\\output.csv', sep=',', index=False,encoding='utf-8')