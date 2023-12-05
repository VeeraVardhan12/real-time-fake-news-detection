from django.shortcuts import render
import pandas as pd
import numpy as np
import googlesearch_py
from sentence_transformers import SentenceTransformer, util
from .forms import factsForm
#from wordcloud import WordCloud
import matplotlib.pyplot as plt 
import io
import urllib, base64
from googlesearch import search
import requests
import inspect
#import requests
import datetime
import time
import spacy
from bs4 import BeautifulSoup
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')
classifier = pipeline('zero-shot-classification')
#model = SentenceTransformer('stsb-roberta-large')
nlp = spacy.load("en_core_web_md")
def home(request):
    if request.method == 'POST':
        form = factsForm(request.POST)
        if form.is_valid():
            facts = form.cleaned_data['facts']
        else:
            facts = factsForm()
        query = facts
        num_results = 4
        """ today = datetime.date.today()
            one_year_ago = today - datetime.timedelta(days=365)

            # Format the dates in the YYYYMMDD format
            start_date = one_year_ago.strftime("%Y%m%d")
            end_date = today.strftime("%Y%m%d")

            # Construct the query string with the date range
            query += f" daterange:{start_date}-{end_date}" 
            """

        # Perform the search
        top_urls=[]
        addme = []
        for j in search(query, stop=num_results, tbs='qdr:y'):
            
            top_urls.append(j)
            
           
            #time.sleep(1)

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        

        
        texts = []
        for i in range(len(top_urls)):
            url = top_urls[i]
            if "youtube" in url:
               # print("Skipping youtube url:",url)
                continue
            if "politifact" in url or "fact-check" in url or "fake-news" in url or "fake" in url or "factcheck" in url:
                addme.append(url)
                fact_check = "We can classify the news as FAKE"
                return render(request,'index.html',{'facts':form,'fine':addme,'fact_check':fact_check})

            response = requests.get(top_urls[i], headers=headers)
            if response.status_code==200:
                soup = BeautifulSoup(response.content, 'html.parser')

                text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
            # texts.append(text)
                if text.strip():
                    texts.append(text)

        LANGUAGE = "english"
        SENTENCES_COUNT = 5
        summaries = []
        for i in range(len(texts)):
            try:
                parser = HtmlParser.from_string(texts[i], None, Tokenizer(LANGUAGE))
                summarizer = LsaSummarizer()
                summarizer.stop_words = [' ']
                summary = summarizer(parser.document, SENTENCES_COUNT)
            # print(f"Summary for website {1}:")
                if summary and any(summary):
                    result = []
                    for i in range(len(summary)):
                        # print(str(summary[i]))
                        result.append(str(summary[i]))
                        str1 = " "
                        str1 = str1.join(result)
                    summaries.append(str1)
               # else:
                #    print("skip")
            except Exception as e:
                #print("error parsing article")
                continue

        similarities = []
       
        for i in range(len(summaries)):
            sentence1 = query
            sentence2 = summaries[i]

            embeddings = [nlp(sentence).vector for sentence in [sentence1, sentence2]]

           
            similarity = cosine_similarity(embeddings)[0][1]

            similarities.append(similarity)

        sentences = np.array(summaries)
        text = ' '.join(sentences)
       


        avg = np.mean(similarities)
        print("score is",avg)

        threshold = 0.50
        
        addme = []
                
        

        if avg < threshold :
            for i in range(0,len(top_urls)):
                fine = top_urls[i]
                addme.append(fine)
            addme = pd.DataFrame(addme, columns=[' ']).set_index([' '])
            print(addme.columns)
            addme = addme.to_html()
            fact_check = "We can classify the news as Fake"

        else:
            for i in range(0,len(top_urls)):
                fine = top_urls[i]
                addme.append(fine)
            addme = pd.DataFrame(addme, columns=[' ']).set_index([' '])
            print(addme.columns)
            addme = addme.to_html()
            
            labels = ['hate speech', 'non-hate speech']
            text = query

            result = classifier(text, labels)

            top_label = result['labels'][0]
            top_score = result['scores'][0]

            if top_label == 'hate speech':
                fact_check = "We can classify the news as Fake"
            else:
                labels = ['Not Profane', 'Profane']
                text = query

                result = classifier(text, labels)

                top_label = result['labels'][0]
                top_score = result['scores'][0]

                if top_label == 'Profane':
                   fact_check = "We can classify the news as Fake"
                else:
                    fact_check = "The News is True"
                        

        return render(request,'index.html',{'facts':form,'fine':addme,'fact_check':fact_check})
    # return render(request,'index.html',{'facts':form,'fine':addme,'fact_check':fact_check,'data':uri})
    
    else:
        form = factsForm()
        return render(request,'index.html',{'facts':form})
        
    