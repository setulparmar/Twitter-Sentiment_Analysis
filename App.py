import matplotlib.pyplot as plt
import pandas as pd
import spacy
import streamlit as st
import string

#from joblib import dump, load
from spacy.lang.pt.stop_words import STOP_WORDS

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB


from tweepy import API
from tweepy import Cursor
from tweepy import OAuthHandler

# File storing the user credentials to access the Twitter API.
import twitter_credentials

import pickle
##################################################3
#My Contribution
from statsmodels.stats.proportion import proportion_confint
import warnings
warnings.filterwarnings("ignore")
# EDA Pkgs
import pandas as pd
import numpy as np
import pandas as pd
import tweepy
import json
from tweepy import OAuthHandler
import re
import textblob
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import openpyxl
import time
import tqdm

#To Hide Warnings
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
# Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
#sns.set_style('darkgrid')
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem import WordNetLemmatizer
import csv, collections
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import csv, collections
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import roc_curve, auc 

STYLE = """
<style>
img {
    max-width: 100%;
}
</style> """

emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

## Defining set containing all stopwords in english.
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

emo_info = {
    # positive emoticons
    ":‑)": " good ",
    ":)": " good ",
    ";)": " good ",
    ":-}": " good ",
    "=]": " good ",
    "=)": " good ",
    ";d": " good ",
    ":d": " good ",
    ":dd": " good ",
    "xd": " good ",
    ":p": " good ",
    "xp": " good ",
    "<3": " love ",

    # negativve emoticons
    ":‑(": " sad ",
    ":‑[": " sad ",
    ":(": " sad ",
    "=(": " sad ",
    "=/": " sad ",
    ":{": " sad ",
    ":/": " sad ",
    ":|": " sad ",
    ":-/": " sad ",
    ":o": " shock "

}

#VOCABULARY_FILENAME = 'better_learned_vocabulary.joblib'
#MODEL_FILENAME = 'Sentiment-NB.pickle'

nltk.download('averaged_perceptron_tagger')
###############################################################################
# Code related to Tweepy.
###############################################################################
class TwitterAuthenticator():
    """
    Class for authenticating our requests to Twitter.
    """
    def authenticate_app(self):
        # The keys are unique identifiers that authenticate the app.
        auth = OAuthHandler(twitter_credentials.CONSUMER_KEY,
                            twitter_credentials.CONSUMER_KEY_SECRET)
        # The tokens allow the app to gain specific access to Twitter data.
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN,
                              twitter_credentials.ACCESS_TOKEN_SECRET)
        return auth


class TwitterClient():
    """
    Class for browsing through tweets via the Twitter API.
    """
    def __init__(self):
        self.auth = TwitterAuthenticator().authenticate_app()
        self.api = API(self.auth)

    def search_tweet(self, query_string):
        cursor = Cursor(self.api.search, q=query_string, lang='en', count=1000,exclude='retweets')
        return cursor.items(100)
    
###############################################################################
def load_sent_word_net():
    sent_scores = collections.defaultdict(list)

    with open("SentiWordNet_3.0.0.txt","r") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='"')

        for line in reader:
            if line[0].startswith("#"):
                continue
            if len(line) == 1:
                continue
            POS, ID, PosScore, NegScore, SynsetTerms, Glos = line
            if len(POS) == 0 or len(ID) == 0:
                continue
            for term in SynsetTerms.split(" "):
                term = term.split('#')[0]
                # print(term)
                term = term.replace("-", " ").replace("_", " ")
                key = "%s/%s" % (POS, term)
                # print(key)
                sent_scores[key].append((float(PosScore), float(NegScore)))
                # print(sent_scores)
        for key, value in sent_scores.items():
            sent_scores[key] = np.mean(value, axis=0)

        return sent_scores

###############################################################################
# Code related to spaCy.
###############################################################################
nlp = spacy.load('pt_core_news_sm')
STOP_WORDS_LIST = list(STOP_WORDS)
PUNCTUATION_CHARACTERS_LIST = list(string.punctuation)






def get_tweets(Topic,Count):
    i=0
    consumer_key = 'It6qdrfzzKYBT0F3B1IQYnpEG'
    consumer_secret = '2UYJ8ys7zQb6z2251xVpMHeloqK09JnnuXyki7LFZ0bTCnIoiy'
    access_token = '1364258204473982978-IrTfDSw2ZvA8IkZqaO30BT0xgBMGv2'
    access_token_secret = 'cPSxtNQBI1BetT71fbMP2B3Ok41UXS8QI1bmPeYYKUyqp'


    # Use the above credentials to authenticate the API.

    auth = tweepy.OAuthHandler( consumer_key , consumer_secret )
    auth.set_access_token( access_token , access_token_secret )
    api = tweepy.API(auth)
    ################################################################
    
    #df1 = pd.DataFrame(columns=["Date","User","IsVerified","Tweet","Likes","RT",'User_location'],dtype=object)
    
    #my_bar = st.progress(100) # To track progress of Extracted tweets
    for tweet in tweepy.Cursor(api.search, q=Topic,count=Count, lang="en",exclude='retweets').items():
        #time.sleep(0.1)
        #my_bar.progress(i)
        df1.loc[i,"Date"] = tweet.created_at
        df1.loc[i,"User"] = tweet.user.name
        df1.loc[i,"IsVerified"] = tweet.user.verified
        df1.loc[i,"Tweet"] = tweet.text
        df1.loc[i,"Likes"] = tweet.favorite_count
        df1.loc[i,"RT"] = tweet.retweet_count
        df1.loc[i,"User_location"] = tweet.user.location
        #df.to_csv("TweetDataset.csv",index=False)
        #df.to_excel('{}.xlsx'.format("TweetDataset"),index=False)   ## Save as Excel
        i=i+1
        if i>Count:
            break
        else:
            pass
# Function to Clean the Tweet.
def clean_tweet(tweet):
    return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([RT])', ' ', tweet.lower()).split())

    
# Funciton to analyze Sentiment
def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

#Function to Pre-process data for Worlcloud
def prepCloud(Topic_text,Topic):
    Topic = str(Topic).lower()
    Topic=' '.join(re.sub('([^0-9A-Za-z \t])', ' ', Topic).split())
    Topic = re.split("\s+",str(Topic))
    stopwords = set(STOPWORDS)
    stopwords.update(Topic) ### Add our topic in Stopwords, so it doesnt appear in wordClous
    ###
    text_new = " ".join([txt for txt in Topic_text.split() if txt not in stopwords])
    return text_new








# Tokenizer used by the CountVectorizer.
def spacy_tokenizer(tweet):
    # Tokenizes tweet and lemmatizes the tokens.
    # It also gets rid of stop words and punctuation characters.
    doc = nlp(tweet)
    tokens = [token.lemma_ for token in doc]
    tokens = [token for token in tokens
              if token not in STOP_WORDS_LIST
              and token not in PUNCTUATION_CHARACTERS_LIST]
    return tokens
###############################################################################


###############################################################################
# Execution code.
###############################################################################

#def get_training_data():
    #df = pd.read_csv('Dataset.csv', sep=';')
    #DATASET_COLUMNS = ['target', 'ids', 'date', 'flag', 'user', 'tweet']
    
    #df= pd.read_csv('dataset1.csv',names=DATASET_COLUMNS,encoding='latin-1')
    #X = df['tweet']
    #y = df['target']
    #return df, train_test_split(X, y, test_size=0.3, random_state=42)


def preprocess(textdata):
    processedText = []
    
    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()
    
    # Defining regex patterns.
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for tweet in textdata:
        tweet = tweet.lower()
        
        # Replace all URls with 'URL'
        tweet = re.sub(urlPattern,' URL',tweet)
        # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])        
        # Replace @USERNAME to 'USER'.
        tweet = re.sub(userPattern,' USER', tweet)        
        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            # Checking if the word is a stopword.
            #if word not in stopwordlist:
            if len(word)>1:
                # Lemmatizing the word.
                word = wordLemm.lemmatize(word)
                tweetwords += (word+' ')
            
        processedText.append(tweetwords)
        
    return processedText


def load_models():
    
    # Load the vectoriser.
    file = open('vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    # Load the LR Model.
    file1 = open('LRmodel.pickle', 'rb')
    LRmodel = pickle.load(file1)
    file1.close()
    
    return vectoriser, LRmodel

def predict(vectoriser,model, text):
    # Predict the sentiment
    #vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
    textdata = vectoriser.transform(text)
    #tfidf_ngrams = TfidfVectorizer(min_df=5, ngram_range=(1, 3))
    #a=tfidf_ngrams.fit(list(textdata))
    sentiment = model.predict(textdata)
    
    #Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
        
    #Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1], ["Negative","Positive"])
    return df



if __name__ == '__main__':
    st.title('Twitter Sentiment Analysis')
    client = TwitterClient()

    #df, (X_train, X_test, Y_train, Y_test) = get_training_data()
    #loaded, pipeline = execute_pipeline(X_train, X_test, Y_train, Y_test)
    #if loaded:
    #    st.write('Machine Learning model loaded.')
    #st.text(f'Model accuracy on test data: {pipeline.score(X_test, Y_test)}')
    vectoriser, LRmodel = load_models()
    # Load the vectoriser.
# =============================================================================
#     file = open('vectoriser-ngram-(1,2).pickle', 'rb')
#     vectoriser = pickle.load(file)
#     file.close()
#     # Load the LR Model.
#     file1 = open('final_model.pickle', 'rb')
#     BNBmodell = pickle.load(file1)
#     file1.close()
# =============================================================================

    keywords = str()
    keywords = str(st.text_input("Enter the topic for Machine Learning Approach"))     
    
    
    keywords2 = str()
    
    
    if len(keywords) > 0 :
        exit
    else :
        keywords2 = str(st.text_input("Enter the topic for Lexican Based Approach"))     
    
    
    
     
    
    if len(keywords) > 0 :
    
        
        # Shows tweets with predictions.
        count = 0
        countp=0
        countn=0
        
        tweets = client.search_tweet(keywords)
        for tweet in tweets:
            #area_key = 'a' + str(count)
            #st.text_area('Fetched tweet', value=tweet.text, key=area_key)
            count += 1
            prediction = predict(vectoriser,LRmodel,[tweet.text])
            #data = pd.DataFrame(data=prediction)
            #st.write('Prediction:',prediction['text'],prediction['sentiment'])
            
            if prediction['sentiment'].all() == "Negative":
                 #st.info('Prediction: negative')
                 countn+=1
            else:
                 #st.info('Prediction: positive')
                 countp+=1
        st.write("Total Positive :"+str(countp) + " And Total Negative:"+str(countn))
        tweets = client.search_tweet(keywords)
        for tweet in tweets:
            #area_key = 'a' + str(count)
            #st.text_area('Fetched tweet', value=tweet.text, key=area_key)
            count += 1
            prediction = predict(vectoriser, LRmodel,[tweet.text])
            #st.write(data)
            st.write('Prediction:',prediction['text'],prediction['sentiment'])
        
        
# =============================================================================
        #st.write(label_dict[sentence.labels[0].value] + ' with ',
        #        sentence.labels[0].score*100, '% confidence')
    
    

    
    if len(keywords2) > 0 :
        df1 = pd.DataFrame(columns=["Date","User","IsVerified","Tweet","Likes","RT","User_location"],dtype=object)
        with st.spinner("Please wait, Tweets are being extracted.."):
            get_tweets(keywords2 , Count=1000)
        st.success('Tweets Extracted!')    
           
    
        # Call function to get Clean tweets
        df1['clean_tweet'] = df1['Tweet'].apply(lambda x : clean_tweet(x))
    
        # Call function to get the Sentiments
        df1["Sentiment"] = df1["Tweet"].apply(lambda x : analyze_sentiment(x))
        
        
        # Write Summary of the Tweets
        st.write("Total Tweets Extracted for Topic '{}' are : {}".format(keywords2,len(df1.Tweet)))
        st.write("Total Positive Tweets are : {}".format(len(df1[df1["Sentiment"]=="Positive"])))
        st.write("Total Negative Tweets are : {}".format(len(df1[df1["Sentiment"]=="Negative"])))
        st.write("Total Neutral Tweets are : {}".format(len(df1[df1["Sentiment"]=="Neutral"])))
        
        # See the Extracted Data : 
        if st.button("See the Extracted Data"):
            #st.markdown(html_temp, unsafe_allow_html=True)
            st.success("Below is the Extracted Data :")
            st.write(df1.head(50))
        
        
        # get the countPlot
        if st.button("Count Plot for Different Sentiments"):
            st.success("Generating A Count Plot")
            st.subheader(" Count Plot for Different Sentiments")
            st.write(sns.countplot(df1["Sentiment"]))
            st.pyplot()
        
        # Piechart 
        if st.button("Pie Chart for Different Sentiments"):
            st.success("Generating A Pie Chart")
            a=len(df1[df1["Sentiment"]=="Positive"])
            b=len(df1[df1["Sentiment"]=="Negative"])
            c=len(df1[df1["Sentiment"]=="Neutral"])
            d=np.array([a,b,c])
            explode = (0.1, 0.0, 0.1)
            st.write(plt.pie(d,shadow=True,explode=explode,labels=["Positive","Negative","Neutral"],autopct='%1.2f%%'))
            st.pyplot()
            
            
        # get the countPlot Based on Verified and unverified Users
        if st.button("Count Plot Based on Verified and unverified Users"):
            st.success("Generating A Count Plot (Verified and unverified Users)")
            st.subheader(" Count Plot for Different Sentiments for Verified and unverified Users")
            st.write(sns.countplot(df1["Sentiment"],hue=df1.IsVerified))
            st.pyplot()
        
        
        ## Points to add 1. Make Backgroud Clear for Wordcloud 2. Remove keywords from Wordcloud
        
        
        # Create a Worlcloud
        if st.button("WordCloud for all things said about {}".format(keywords2)):
            st.success("Generating A WordCloud for all things said about {}".format(keywords2))
            text = " ".join(review for review in df1.clean_tweet)
            stopwords = set(STOPWORDS)
            text_newALL = prepCloud(text,keywords2)
            wordcloud = WordCloud(stopwords=stopwords,max_words=800,max_font_size=70).generate(text_newALL)
            st.write(plt.imshow(wordcloud, interpolation='bilinear'))
            st.pyplot()
        
        
        #Wordcloud for Positive tweets only
        if st.button("WordCloud for all Positive Tweets about {}".format(keywords2)):
            st.success("Generating A WordCloud for all Positive Tweets about {}".format(keywords2))
            text_positive = " ".join(review for review in df1[df1["Sentiment"]=="Positive"].clean_tweet)
            stopwords = set(STOPWORDS)
            text_new_positive = prepCloud(text_positive,keywords2)
            #text_positive=" ".join([word for word in text_positive.split() if word not in stopwords])
            wordcloud = WordCloud(stopwords=stopwords,max_words=800,max_font_size=70).generate(text_new_positive)
            st.write(plt.imshow(wordcloud, interpolation='bilinear'))
            st.pyplot()
        
        
        #Wordcloud for Negative tweets only       
        if st.button("WordCloud for all Negative Tweets about {}".format(keywords2)):
            st.success("Generating A WordCloud for all Positive Tweets about {}".format(keywords2))
            text_negative = " ".join(review for review in df1[df1["Sentiment"]=="Negative"].clean_tweet)
            stopwords = set(STOPWORDS)
            text_new_negative = prepCloud(text_negative,keywords2)
            #text_negative=" ".join([word for word in text_negative.split() if word not in stopwords])
            wordcloud = WordCloud(stopwords=stopwords,max_words=800,max_font_size=70).generate(text_new_negative)
            st.write(plt.imshow(wordcloud, interpolation='bilinear'))
            st.pyplot()
        
        
        
        
