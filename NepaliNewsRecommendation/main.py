from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import re
import difflib
import pandas as pd
import numpy as np
import nltk
from nepali_stemmer.stemmer import NepStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')


app = FastAPI(encoding="UTF-8")
templates = Jinja2Templates(directory="templates")

# stemmer object
nepstem = NepStemmer()

# adding new words to the stopwords for nepali language
stop_words = set(stopwords.words('nepali'))
custom_stopwords = ['छ', 'हो', 'ले', 'को']
stop_words.update(custom_stopwords)

cosine_similarities_matrix = np.load('cosine_similarities.npy', allow_pickle = True)
data_matrix = np.load('SampleData.npy', allow_pickle = True)
all_news = data_matrix[:, -1]
columns = ['content', 'heading', 'main_topic', 'complete_text']
df = pd.DataFrame(data_matrix, columns = columns)


# ------------------------------------- DATA CLEANING CODE START -------------------------------
def lower_order(text):
    return text

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def rm_html(text):
    return re.sub(r'<[^>]+>', '', text)

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r' ', string)

def remove_unwanted_characters_nepali(document):
    # remove user mentions
    document = re.sub("@[^\s]+", " ", document)
    # remove hashtags
    document = re.sub("#[^\s]+", "", document)
    # remove punctuation (except for Nepali Unicode characters)
    document = re.sub("[^\u0900-\u097F0-9A-Za-z\s]", "", document)
    # remove emojis
    document = remove_emoji(document)
    # remove double spaces
    document = document.replace('  ', "")
    document = re.sub(r"[^\w\s]", "", document)
    return document.strip()

def rm_whitespaces(text):
    return re.sub(r' +', ' ', text)

def remove_punctuations_nepali(sentence):
    punctuations = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~।॥"""
    return re.sub('[' + re.escape(punctuations) + ']', '', sentence)

def remove_stopwords_nepali(text):
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def nepali_stemming(text):
  return nepstem.stem(text)

def remove_nepali_numbers(text):
    # Regular expression pattern to match numeric characters in Nepali script
    pattern = r'[०-९]'
    # Remove numeric characters using regex substitution
    text_without_numbers = re.sub(pattern, '', text)
    return text_without_numbers

def tokenize(text):
    return word_tokenize(text)

def custom_cleaning_pipeline(string):   
    string = lower_order(string)
    string = remove_urls(string)
    string = rm_html(string)
    string = remove_emoji(string)
    string = remove_unwanted_characters_nepali(string)
    string = remove_punctuations_nepali(string)
    string = remove_nepali_numbers(string)
    string = rm_whitespaces(string)
    string = remove_stopwords_nepali(string)
    string = tokenize(string)
    string = nepali_stemming(" ".join(string))
    return string

# ------------------------------------- DATA CLEANING CODE END -------------------------------

def recommend_me_a_news(text, k):
    news = custom_cleaning_pipeline(text)
    detecting_similar_news = difflib.get_close_matches(news, all_news)[0]
        
    # finding the index of the news
    index_of_the_news = df[df.complete_text == detecting_similar_news].index.values[0]
    
    # Calculating the similar news' indexes and their similarity values in a tuple inside another list
    similarity_score = list(enumerate(cosine_similarities_matrix[index_of_the_news]))
    
    # Sorting the previous List in accordance to their similarity score using Scored Function
    sorted_similar_news = sorted(similarity_score, key = lambda x : x[1], reverse = True) 
    sorted_similar_news.remove(sorted_similar_news[0]) # the first element is the same as the input text
    
    # seleting top k similarities
    sorted_similar_news = sorted_similar_news[:k]
    
    indices = [i[0] for i in sorted_similar_news]
    similarities_values = [i[1] for i in sorted_similar_news]
    
    return data_matrix[indices], similarities_values   


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    with open("templates/webapp.html", "r") as file:
        content = file.read()
        return templates.TemplateResponse("webapp.html", {"request": request, "entire_text": "", "input_text": ""})    

@app.post("/recommend")
async def process_text(request: Request):
    form_data = await request.form()
    input_text = form_data["nepaliText"]  
    k = form_data["numNews"]  
    
    recommendations, similarities_values = recommend_me_a_news(input_text, int(k))
    entire_text = ""

    for i, row in enumerate(recommendations):
        content = row[0]
        heading = row[1]
        category = row[2]
        entire_text += "<b>{}. {} ".format(i+1, heading) +", News Category: "+category+ ", Cosine Similarity: {}".format(round(similarities_values[i], 5)) +"</b><br>"
        entire_text += content + "<br><br>"

    return templates.TemplateResponse("webapp.html", {"request": request, "entire_text": entire_text, "input_text": input_text})
