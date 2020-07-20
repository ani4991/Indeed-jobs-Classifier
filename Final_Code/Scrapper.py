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
import urllib.parse

## Reading the input company name file

company_df = pd.read_csv("C:\\Users\\ani49\\Data_Science_Case_Studies\\Enlyft\\enlyft_datascience_sample.csv")
print(company_df.shape)

## Converting to suitable list format

company_list = company_df['Company Name'].values.tolist()
print(len(company_list))

## web scrapper function - for all the companies in the list, this function scrapes 4 features:

# 1. Job title
# 2. Company
# 3. Location
# 4. Summary

## declaring the main dictionary variable that stores alla the scrapped data
list_of_companies_df_dict = {}


for i in company_list:
    
    #name = i.split()
    #query = ""
    page = 0
    with open('C:\\Users\\ani49\\Data_Science_Case_Studies\\Enlyft\\scrape_results-2.csv', 'a', encoding='utf-8', newline='') as f_output:
        csv_print = csv.writer(f_output)

        file_is_empty = os.stat('C:\\Users\\ani49\\Data_Science_Case_Studies\\Enlyft\\scrape_results-2.csv').st_size == 0
        if file_is_empty:
            csv_print.writerow(['Job_title', 'Company', 'Location', 'Summary'])

        #for j in range(len(name)-1):
            #query += name[j] + '+'

        #query += name[len(name)-1]
        
        URL = 'https://www.indeed.com/jobs?q={}&start={}'.format(urllib.parse.quote_plus(i),page)
        print("URL connected is: ", URL)
        request = requests.get(URL)
        
        while(request.status_code == 200):

            soup = BeautifulSoup(urllib.request.urlopen(URL).read(), 'html.parser')


            for jobs in soup.find_all(class_='result'):

                try:
                    title = jobs.div.text.strip()
                except Exception as e:
                    title = None
                #print('Job_title:', title)


                try:
                    company = jobs.span.text.strip()
                except Exception as e:
                    company = None
                #print("Company:", company)


                try:
                    location = jobs.find('span', class_='location').text.strip()
                except Exception as e:
                    location=None
                #print("Location:", location)


                try:
                    summary = jobs.find('div', class_='summary').text.strip()
                except Exception as e:
                    summary=None
                #print('Summary:', summary)


                csv_print.writerow([title, company, location, summary])

                #print("---------------------------")



                time.sleep(0.1)
            
            page += 10
            URL = 'https://www.indeed.com/jobs?q={}&start={}'.format(urllib.parse.quote_plus(i),page)
            request = requests.get(URL)
        
    df = pd.read_csv('C:\\Users\\ani49\\Data_Science_Case_Studies\\Enlyft\\scrape_results-2.csv')
    list_of_companies_df_dict[i] = df
    os.remove('C:\\Users\\ani49\\Data_Science_Case_Studies\\Enlyft\\scrape_results-2.csv')


## finally saving the intermediate dataframe object to respective company data as csv files 

for key,value in list_of_companies_df_dict.items():
    print(key)
    df.to_csv('C:\\Users\\ani49\\Data_Science_Case_Studies\\Enlyft\\new\\' + key + '.csv', sep=',', index=False,encoding='utf-8')
