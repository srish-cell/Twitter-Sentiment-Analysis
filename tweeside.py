import time
import pickle
import datetime
import pandas as pd
import numpy as np
import preprocessors
import tweepy
import config
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier as xgb
import statsmodels.api as sm
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score,classification_report
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import spacy
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import matplotlib.pyplot as plt

client = tweepy.Client(
    consumer_key=config.API_KEY,
    consumer_secret=config.API_SECRET,
    access_token=config.ACCESS_TOKEN,
    access_token_secret=config.ACCESS_SECRET,
    bearer_token=config.BEARER_TOKEN
)

query = "Covid-19 -is:retweet"
start_time=datetime.datetime(2023,2,9,3,30,0)
end_time=datetime.datetime(2023,2,10,6,0,0)
tweet1=tweepy.Paginator(client.search_recent_tweets,query = "Covid",max_results = 100).flatten(limit = 500)
df = pd.DataFrame(data = [tweet.text for tweet in tweet1], columns=['Tweets'])
df['clean_tweet'] = df['Tweets'].apply(lambda x: x.lower())
df["stemmed_data"] = preprocessors.pstem_clean_text(df)
df['lemmatized_text'] = preprocessors.lemmatize_clean_text(df)
df['sub'],df['pol'],df['sen'] = preprocessors.sentiment_polarity(df,'lemmatized_text')
for tweet in df['lemmatized_text']:
  print(tweet) 
      
cv = CountVectorizer(lowercase= False) 
text_vector = cv.fit_transform(df.lemmatized_text.values)
x = text_vector
y1 = np.array(df.iloc[:,6:])
y = y1.reshape(-1,1)
X_train, X_test, y_train, Y_test = train_test_split(x, y,train_size=0.2,stratify=y,random_state=42)
le = LabelEncoder()
y_train = le.fit_transform(y_train) 
Y_test = le.fit_transform(Y_test) 
Y_test1=np.array(Y_test)
Y_test =Y_test1.reshape(-1,1)
y_train1=np.array(y_train)
y_train=y_train1.reshape(-1,1)

model = xgb(objective= 'multi:softprob',learning_rate = 0.5001 ,
                 n_estimators = 100,max_depth = 6,seed=42   )
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy is :",accuracy)
print(classification_report(Y_test, y_pred))
df = df.set_index("timestamp")
sia = SentimentIntensityAnalyzer()
positive_count = 0
negative_count = 0
neutral_count = 0
df["sentiment"] = df["Tweets"].apply(lambda x: sia.polarity_scores(x)["compound"])
sentiment_ts = df["sentiment"]
model1 = sm.tsa.ARIMA(sentiment_ts, order=(1, 0, 1))
arima_fit = model1.fit()
string = pd.Series(df["lemmatized_text"]).str.cat(sep=' ')
plt.plot(sentiment_ts, label="Sentiment")
plt.plot(arima_fit.predict(), label="ARIMA Prediction")
plt.legend()
plt.show()
stopwords = set(STOPWORDS)
stopwords.update(["COVID","CORONA","COVID-19","Covaxin","CoronaVirus","Lockdown","Quarantine"])
wordcloud = WordCloud(width=1600, stopwords=stopwords,height=800,max_font_size=200,max_words=50,collocations=False, background_color='black').generate(string)
plt.figure(figsize=(40,30))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


