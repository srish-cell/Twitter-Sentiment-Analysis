from flask import Flask,render_template,request,jsonify
import tweepy
from textblob import TextBlob
import tweepymain as tpy
import config


#---------------------------------------------------------------------------

consumer_key = 't4jY0Jy7ochxHXCz1P0wOhL91'
consumer_secret = 'B7tWUupoLGtS59FXaV935tqSrkVDVFOrzu73J3d7giWBHOOisN'

access_token = '1480088496811302912-PbS0ixiZp00IEE9ku6uPQNgvKYAJHn'
access_token_secret = 'HY7wrRXtiHdCt1YOzDCWR29GoN4VgvcPxnaOCYz847ykX'

auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#-------------------------------------------------------------------------

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/search",methods=["GET","POST"])
def search():
    search_tweet = request.form.get("search_query")
    tweets=tpy.qry(search_tweet)
    t = []
        # t.append(tweet.full_text)
    return render_template('out.html')

 
app.run()