from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import re
import numpy as np
import torch
import torch.nn as nn
import nltk
import pickle
from nepali_stemmer.stemmer import NepStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fastapi.templating import Jinja2Templates
nltk.download('stopwords')
nltk.download('punkt')


app = FastAPI()
templates = Jinja2Templates(directory="templates")

# stemmer object
nepstem = NepStemmer()

# adding new words to the stopwords for nepali language
stop_words = set(stopwords.words('nepali'))
custom_stopwords = ['छ', 'हो', 'ले', 'को']
stop_words.update(custom_stopwords)


# Load the pickle file that contains the vocab dictionary and index positions
with open('vocab_dictionary.pkl', 'rb') as file:
    converting_integers_and_words = pickle.load(file)

# Load the pickle file that contains the label encoded class names
with open('class_info.pkl', 'rb') as file:
    class_info = pickle.load(file)

''' ------------------------------------- MODEL CODE START -------------------------------------------------'''
class LSTM(nn.Module):
    def __init__(self, vocab_size, output_size, hidden_size=512, embedding_size=300, n_layers=3, dropout=0.4):
        super(LSTM, self).__init__()

        # Define an embedding layer that maps each token to a dense vector of embedding_size
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)

        # Define an LSTM layer with hidden_size hidden units, n_layers layers, and a dropout rate of dropout
        self.lstm_layer = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout, batch_first=True)

        # Define a dropout layer with dropout probability of dropout
        self.dropout_layer = nn.Dropout(p=dropout)

        # Define a linear layer that maps the output of the LSTM to the output_size
        self.fully_connected_laeyer = nn.Linear(hidden_size, output_size)

        # Define a sigmoid activation function
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, input_seq):
        
        # Convert the input to a LongTensor
        input_seq = input_seq.long()

        # Embed the input sequence to a sequence of dense vectors of embedding_size
        input_seq = self.embedding_layer(input_seq)

        # Feed the embedded sequence through the LSTM layer
        output, _ = self.lstm_layer(input_seq)

        # Select only the last output of the LSTM as the final output
        output = output[:, -1, :]

        # Apply dropout to the output
        output = self.dropout_layer(output)

        # Feed the output through the linear layer to get the logits
        output = self.fully_connected_laeyer(output)

        return output
    
    # define training device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# model hyperparamters
vocab_size = 453729
output_size = 18
embedding_size = 300
hidden_size = 512
n_layers = 2
dropout=0.2
weight_decay=None

# model initialization
model = LSTM(vocab_size, output_size, hidden_size, embedding_size, n_layers, dropout)
model = model.to(device)
model.load_state_dict(torch.load('LSTM_Original.pth'))
model.eval()

# ------------------------------------ MODEL CODE END ---------------------------------------------

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

# padding sequences
def padding_sequences(cleaned_text, padding_value, max_length_of_sequence=256):
    '''
    The code below will create a 2D numpy array called new_feature with dimensions (len(reviews), seq_length), filled with the padding ID pad_id. 
    Therefore, we can use this 2d array to replace the corresponidng sequences at respective indexes whereas unused elements for sequences shorter than 256 will be padded.
    '''
    new_feature = np.full((len(cleaned_text), max_length_of_sequence), padding_value, dtype=int)
    '''
    pads each sequence with the padding ID pad_id up to the desired seq_length. The code first converts the current sequence to a
      numpy array using np.array(row), then takes the first seq_length elements (if the sequence is longer than seq_length), and assigns 
      this to the corresponding row in features. This way, sequences shorter
      than seq_length are padded with the pad_id at the end of the sequence, and longer sequences are truncated to seq_length.
    '''
    for i, row in enumerate(cleaned_text):
        new_feature[i, :len(row)] = np.array(row)[:max_length_of_sequence]
    return new_feature

# function to make individual predictions
def predict_sentiment(text):
    text = custom_cleaning_pipeline(custom_cleaning_pipeline(text))
    text = [[converting_integers_and_words[word] for word in text.split() if word in converting_integers_and_words.keys()]]
    text = padding_sequences(text, padding_value=converting_integers_and_words['<PAD>'])
    text_tensor = torch.tensor(text).to(device)
    prediction = model(text_tensor).cpu().detach().numpy()
    # Applying softmax activation
    softmax_output = np.exp(prediction) / np.sum(np.exp(prediction))
    return softmax_output


''' ------------------------------------------ REMAINING FASTAPI CODE----------------------------------------'''
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    with open("templates/webpage.html", "r") as file:
        content = file.read()
        return templates.TemplateResponse("webpage.html", {"request": request, "original_text": "", "cleaned_text": "",\
                                                           "Predicted_Category":"", "Predicted_Probability":""})    

@app.post("/process")
async def process_text(request: Request):
    form_data = await request.form()
    nepali_text = form_data["nepali_text"]
    cleaned_nepali_text = custom_cleaning_pipeline(nepali_text)
    print("Original Nepali Text:", nepali_text)
    print("Cleaned Nepali Text:", cleaned_nepali_text)
    prediction = predict_sentiment(cleaned_nepali_text)
    print("All news categories: ", class_info)
    predicted_class = class_info[np.argmax(prediction)]
    predicted_probability = str(np.max(prediction))
    print("Predicted Class: ", predicted_class)
    print("Predicted Probability: ", predicted_probability)
    
    return templates.TemplateResponse("webpage.html", {"request": request, "original_text": nepali_text, "cleaned_text": cleaned_nepali_text,\
                                                       "Predicted_Category": predicted_class, "Predicted_Probability":predicted_probability})