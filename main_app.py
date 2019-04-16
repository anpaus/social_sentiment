import dash
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import plotly 
import plotly.plotly as py
import plotly.graph_objs as go
import sqlite3
import pandas as pd
from collections import Counter
import string
import regex as re
import time
import pickle
import nltk
import re
# set chdir to current dir
import sys
import os
import tweepy
import json
import base64

import numpy as np
from wordcloud import WordCloud, STOPWORDS

sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)))
os.chdir(os.path.realpath(os.path.dirname(__file__)))

from cache import cache
from config import stop_words


# source:  https://pythonprogramming.net/basic-flask-website-tutorial/
app = dash.Dash(__name__)
server = app.server


# it's ok to use one shared sqlite connection
# as we are making selects only, no need for any kind of serialization as well
conn = sqlite3.connect('twitter_wloc.db', check_same_thread=False)

punctuation = [str(i) for i in string.punctuation]

sentiment_colors = {-1:"#d01111",
                    -0.3:"#f7a1a1",
                     0:"#fff0cc",
                     0.3:"#b3de87",
                     1:"#80c837",}



app_colors = {
    'background': '#0C0F0A',
    'text': '#FFFFFF',
    'sentiment-plot':'#41EAD4',
    'volume-bar':'#FBFC74',
    'someothercolor':'#FFE4C4',
}


app.layout = html.Div([
        
       html.Div(className='container-fluid', children=[html.Br(), html.H2 ('real time twitter sentiment', style={'textAlign': 'center', 'color':app_colors['someothercolor']}),   
                                                  html.H4 ('Each refresh may take ~1min   |   English tweets only', style={'textAlign': 'center', 'color':app_colors['someothercolor']}),   
                                                  html.H4('sentiment search for:', style={'color':app_colors['someothercolor']}),
                                                  dcc.Input(id='sentiment_term', value='twitter', type='text', style={'color':app_colors['text']}),  
                                                  html.Div(['example'], id='input-div', style={'display': 'none'}),  #Added for button click. I do not display to hide the text box.
                                                  html.Button('submit', id="submit-button")               #Added for button click
                                                        ], style={'width':'95%','margin-left':10,'margin-right':10, 'max-width':50000}),
     
      html.Div(className='row', children= [
                  dcc.Input(id='trending_id', type='text', style={'display': 'none'}), 
                  html.H4('recently trending: ', className='col s12 m6 l6', style={'color':app_colors['someothercolor']}), 
                  html.H4('related terms trending: ', className='col s12 m6 l6', style={'color':app_colors['someothercolor']}),
                  html.H6(id='trending_topics', className='col s12 m6 l6', style={'color':app_colors['text'], "word-wrap":"break-word"} ),
                  #html.Img(id='wordcloud', src='data:image/png;base64,{}'.format(encoded_image.decode()), className='col s12 m6 l6', style={'width': '500px'})
                  html.Img(id='wordcloud', className='col s12 m6 l6', style={'width': '500px'})
                  #html.H6(id='rel_trending_topics', className='col s12 m6 l6', style={'color':app_colors['text'], "word-wrap":"break-word"})
                                        ]),

       html.Div(className='row', children=[html.Div(dcc.Graph(id='live-graph', animate=False), className='col s12 m6 l6'),
                                            html.Div(dcc.Graph(id='historical-graph', animate=False), className='col s12 m6 l6')]),

       html.Div(className='row', children= [html.Div(dcc.Graph(id='choropleth_map', style={"height": "100%", "width": "100%"}, config=dict(displayModeBar=True)), className='col s12 m6 l6'),
                                             html.Div(dcc.Graph(id='sentiment-pie', animate=False), className='col s12 m6 l6')]),

       html.Div(className='row', children= [html.Div(id="recent-tweets-table", className='col s12 m12 l12'),
                                            
                                             html.H2('   ', style={'color':"#000000"}, className='col s12 m6 l12'),
                                             html.Br(), html.Br(), html.Br(),  html.Br(),  html.Br(),
                                             html.H6('2019 - Contact: gencozgur@gmail.com', style={'textAlign': 'center', 'color':"#FFEFE8"}, className='col s12 m6 l12')]),

       
       
       dcc.Link('Project idea and original code from "Sentdex"', href='https://github.com/Sentdex/socialsentiment'),

       dcc.Interval(
            id='live-graph-update',
            interval=20*1000, n_intervals=0, max_intervals= 5
        ),
       dcc.Interval(
            id='historical-update',
            interval=100*1000, n_intervals=0, max_intervals= 0
        ),

       dcc.Interval(
           id='wordcloud-update',
          interval=120*1000, n_intervals=0, max_intervals= 0
        ),
 
       dcc.Interval(
            id='recent-table-update',
            interval=30*1000, n_intervals=0, max_intervals= 15
        ),

       dcc.Interval(
            id='sentiment-pie-update',
            interval=150*1000, n_intervals=0, max_intervals= 0
        ),        
        
       dcc.Interval(
            id='choropleth_map-update',
            interval=170*1000, n_intervals=0, max_intervals= 0
        ),
        
                 ], style={'backgroundColor': app_colors['background'], 'margin-top':'-30px', 'height':'3000px',}
                )


# returns a choropleth map figure based on states codes and avg sentiments source file
def choropleth_map(df, sentiment_term):
    
    sentiment_term = removeSpecialChar(sentiment_term)
    for col in df.columns:
        df[col] = df[col].astype(str)

    scl = [[0.0, 'rgb(209,17,17)'],[0.20, 'rgb(204, 65, 65)'],[0.40, 'rgb(228,255,172)'],\
            [0.60, 'rgb(118,232,168)'],[0.80, 'rgb(56, 209, 123)'], [1.0, 'rgb(69,163,32)']]

    df['text'] = df['state_code'] + '<br>' +\
        'Avg sentiment: ' + df['sentiment'] + '<br>' +\
        '# of tweets: ' + df['nb_tweets']

    data = [ dict(
            type='choropleth',
            colorscale = scl,
            autocolorscale = False,
            locations = df['state_code'],
            z = df['sentiment'].astype(float),
            locationmode = 'USA-states',
            text = df['text'],
            marker = dict(
                line = dict (
                            color = 'rgb(255,255,255)',
                            width = 2
                          ) ),
                colorbar = dict(
                title = "Twitter sentiment")
           ) ]

    layout = dict(
            title = 'Social media sentiment of "' + sentiment_term + '" by States<br>(Hover for breakdown)',
            geo = dict(
                scope='usa',
                projection=dict( type='albers usa' ),
                showlakes = True,
                lakecolor = 'rgb(255, 255, 255)'),
                    )
    return dict(data=data, layout=layout)


def removeSpecialChar (raw_text):
    # a = '!!I am now m@@@king $ %^&$ a test #@!) to remove special ()$* characters!!! ok?'
    #raw_text = raw_text.split()
    for k in raw_text.split("\n"):
        #print(re.sub(r"[^a-zA-Z0-9]+", ' ', k))
        return(re.sub(r"[^a-zA-Z0-9]+", ' ', k))   #Remove all special characters.



@app.callback(
    Output(component_id='trending_topics', component_property='children'),
    [Input(component_id='trending_id', component_property='value')])
def update_output_div(input_value):
    t12 = time.time()
    #print('trending topics app callback e girdim:  ', t1)

    consumer_key = 'XXXXXXXXX'
    consumer_secret = 'XXXXXXXXXX'
    access_token = 'XXXXXXXXXXXX'
    access_token_secret = 'XXXXXXXXXXXXXXXXXXX'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    # Where On Earth ID for USA is 23424977.
    US_WOE_ID = 23424977
    US_trends = api.trends_place(US_WOE_ID)
    trends = json.loads(json.dumps(US_trends, indent=1))

    #print(trends)
    data = []
    for trend in trends[0]["trends"]:
        data.append(trend["name"])

    myTrends = " | ".join(str(elem) for elem in data[:25])
    t22 = time.time()
    print('trending topics app callback bitirdim:  ', t22 - t12)
    return myTrends


@app.callback(Output('wordcloud', 'src'),
             [Input('input-div', 'children'),
             Input('wordcloud-update', 'n_intervals')])
def update_wordcloud(sentiment_term, n_intervals):
    t13 = time.time()
    #print('word cloud app call back e girdim:  ', t1)
    access_token = 'XXXXXXXXXXXXXX'
    access_token_secret = 'XXXXXXXXXXXXXX'
    consumer_key = 'XXXXXXXXXXXXXX'
    consumer_secret = 'XXXXXXXXXXXXXX'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    conn = sqlite3.connect('twitter_wloc.db', check_same_thread=False)
    sql_term = "SELECT unix, tweet FROM sentiment WHERE tweet LIKE '%"+sentiment_term+"%' ORDER BY id DESC LIMIT 1000"

    df = pd.read_sql(sql_term, conn)

    # join tweets to a single string
    raw_words = ' '.join(df['tweet']).lower()

    #remove reoccuring words in a string
    noduplicate_words = re.sub(r'\b(\w+)( \1\b)+', r'\1', raw_words)

    # remove URLs, RTs, and twitter handles
    no_urls_no_tags = " ".join([word for word in noduplicate_words.split()
                                if 'http' not in word
                                    and not word.startswith('@')
                                    and word != 'RT'
                                    and word !='rt'
                                    and word !=sentiment_term
                                    and word !=sentiment_term.lower()
                                    and word !=sentiment_term.upper()
                                ])

    no_unicode = re.sub(r"\\[a-z][a-z]?[0-9]+", '', no_urls_no_tags)
    if len(no_unicode) < 20:
        no_unicode = 'Please search for a more popular word.'

    wordcloud = WordCloud(stopwords=STOPWORDS, 
                        background_color='black',
                        width=2800,
                        height=800,
                        max_words = 50
                        ).generate(no_unicode)

    wordcloud.to_file('wordcloud.png')
    image_path = 'wordcloud.png' # Have an image saved of a wordcloud in the main project folder
    #html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))

    encoded_image = base64.b64encode(open(image_path, 'rb').read())
    t23 = time.time()
    print('word cloud app call back bitirdim:  "{}" '.format(sentiment_term), t23 - t13)
    return 'data:image/png;base64,{}'.format(encoded_image.decode())
    

MAX_DF_LENGTH = 100


def df_resample_sizes(df, maxlen=MAX_DF_LENGTH):
    df_len = len(df)
    resample_amt = 100
    vol_df = df.copy()
    vol_df['volume'] = 1

    ms_span = (df.index[-1] - df.index[0]).seconds * 1000
    rs = int(ms_span / maxlen)

    df = df.resample('{}ms'.format(int(rs))).mean()
    df.dropna(inplace=True)

    vol_df = vol_df.resample('{}ms'.format(int(rs))).sum()
    vol_df.dropna(inplace=True)

    df = df.join(vol_df['volume'])

    return df

# make a counter with blacklist words and empty word with some big value - we'll use it later to filter counter
stop_words.append('')
blacklist_counter = Counter(dict(zip(stop_words, [1000000]*len(stop_words))))

# complie a regex for split operations (punctuation list, plus space and new line)
split_regex = re.compile("[ \n"+re.escape("".join(punctuation))+']')

POS_NEG_NEUT = 0.1

def quick_color(s):
    # except return bg as app_colors['background']
    if s >= POS_NEG_NEUT:
        # positive
        return "#76e8a8"
    elif s <= -POS_NEG_NEUT:
        # negative:
        return "#cc4141"

    else:
        return app_colors['background']

def generate_table(df, max_rows=20):
    return html.Table(className="responsive-table",
                      children=[
                          html.Thead(
                              html.Tr(
                                  children=[
                                      html.Th(col.title()) for col in df.columns.values],
                                  style={'color':app_colors['text']}
                                  )
                              ),
                          html.Tbody(
                              [
                                  
                              html.Tr(
                                  children=[
                                      html.Td(data) for data in d
                                      ], style={'color':app_colors['text'],
                                                'background-color':quick_color(d[2])}
                                  )
                               for d in df.values.tolist()])
                          ]
    )


def pos_neg_neutral(col):
    if col >= POS_NEG_NEUT:
        # positive
        return 1
    elif col <= -POS_NEG_NEUT:
        # negative:
        return -1

    else:
        return 0
    

#This part is added as a work around to use button click. Now you update the div only on button click. 
#Though the Graph always uses the the div content to query your database (either when the interval triggers or the div changes):
@app.callback(Output('input-div', 'children'),  
              [Input('submit-button', 'n_clicks')],
              state=[State(component_id='sentiment_term', component_property='value')])
def update_div(n_clicks, sentiment_term):
    sentiment_term = removeSpecialChar(sentiment_term)    
    return sentiment_term



@app.callback(Output('recent-tweets-table', 'children'),
              [Input('input-div', 'children'),
               Input('recent-table-update', 'n_intervals')])        
def update_recent_tweets(sentiment_term, n_intervals):
    t14 = time.time()
    #print('recent tweets table app call back e girdim  ', t1)
    try: 
        sentiment_term = removeSpecialChar(sentiment_term)

        if sentiment_term:
            sql_term = "SELECT sentiment.* FROM sentiment WHERE tweet LIKE '%"+sentiment_term+"%' AND followers_count > 200 ORDER BY id DESC LIMIT 15"
            df = pd.read_sql(sql_term, conn)
        else:
            df = pd.read_sql("SELECT * FROM sentiment ORDER BY id DESC, unix DESC LIMIT 15", conn)

        df['date'] = pd.to_datetime(df['unix'], unit='ms')
        df = df.drop(['unix','id'], axis=1)
        df = df[['date','tweet', 'sentiment', 'followers_count']]

        t24 = time.time()
        print('recent tweets table app call back bitirdim:  "{}" '.format(sentiment_term), t24 - t14)

        return generate_table(df, max_rows=20)
    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')

@app.callback(Output("choropleth_map", "figure"),
             [Input('input-div', 'children'),
                 #Input(component_id='sentiment_term', component_property='value'),
              Input('choropleth_map-update', 'n_intervals')]) 
def map_callback(sentiment_term, n_intervals):
    #df = pd.read_csv(r'C:\Users\genco\socialsentiment\socialsentiment_export.csv')
    
    t15 = time.time()
    #print('cholop app callback fonk girdim:  ', t1)
    
    sentiment_term = removeSpecialChar(sentiment_term)
    if sentiment_term:
        sql_term = "SELECT state_code, AVG(sentiment) AS sentiment, COUNT(tweet) AS nb_tweets FROM sentiment WHERE tweet LIKE '%"+sentiment_term+"%' GROUP BY state_code"
        df = pd.read_sql(sql_term, conn)
    #print(df)
    
    t25 = time.time()
    print('cholop app callback bitirdim:  "{}" '.format(sentiment_term), t25 - t15)

    return choropleth_map(df, sentiment_term)


@app.callback(Output('live-graph', 'figure'),
              [Input('input-div', 'children'),
             # [Input(component_id='sentiment_term', component_property='value'),  #I commented out this line after I put the submit button.
               Input('live-graph-update', 'n_intervals')])
def update_graph_scatter(sentiment_term, n_intervals):
    
    t17 = time.time()
    #print('live graph app call back fon na girdim:  ', t1)

    try:
        
        if sentiment_term:
            sql_term = "SELECT unix, sentiment FROM sentiment WHERE tweet LIKE '%"+sentiment_term+"%' ORDER BY id DESC LIMIT 800"
            df = pd.read_sql(sql_term, conn)
        else:
            df = pd.read_sql("SELECT * FROM sentiment ORDER BY id DESC, unix DESC LIMIT 800", conn)
        
        df.sort_values('unix', inplace=True)
        df['date'] = pd.to_datetime(df['unix'], unit='ms')
        df.set_index('date', inplace=True)
        init_length = len(df)
        df['sentiment_smoothed'] = df['sentiment'].rolling(int(len(df)/5)).mean()
        df = df_resample_sizes(df)
        X = df.index
        Y = df.sentiment_smoothed.values
        Y2 = df.volume.values
        data = plotly.graph_objs.Scatter(
                x=X,
                y=Y,
                name='Sentiment',
                mode= 'lines',
                yaxis='y2',
                line = dict(color = (app_colors['sentiment-plot']),
                            width = 4,)
                )

        data2 = plotly.graph_objs.Bar(
                x=X,
                y=Y2,
                name='Volume',
                marker=dict(color=app_colors['volume-bar']),
                )

        t27 = time.time()
        print('live graph app call back fonk bitirdim: "{}" '.format(sentiment_term), t27 - t17)

        return {'data': [data,data2],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                                          yaxis=dict(range=[min(Y2),max(Y2*4)], title='Volume', side='right'),
                                                          yaxis2=dict(range=[min(Y),max(Y)], side='left', overlaying='y',title='sentiment'),
                                                          title='Live sentiment for: "{}"'.format(sentiment_term),
                                                          font={'color':app_colors['text']},
                                                          plot_bgcolor = app_colors['background'],
                                                          paper_bgcolor = app_colors['background'],
                                                          showlegend=False)}

    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')


@app.callback(Output('historical-graph', 'figure'),
              [Input('input-div', 'children'),
               #Input(component_id='sentiment_term', component_property='value'),
                Input('historical-update', 'n_intervals')])  
def update_hist_graph_scatter(sentiment_term, n_intervals):
    
    t18 = time.time()
    #print('historical app call back e girdim:  ', t1)
    
    try:
        if sentiment_term:
            sql_term = "SELECT unix, sentiment FROM sentiment WHERE tweet LIKE '%"+sentiment_term+"%' ORDER BY id DESC LIMIT 20000"
            df = pd.read_sql(sql_term, conn)
        else:
            df = pd.read_sql("SELECT * FROM sentiment ORDER BY id DESC, unix DESC LIMIT 1000", conn)

        df.sort_values(by = ['unix'], inplace=True)  # does this re-sort time form begin to today or vica versa??
        df['date'] = pd.to_datetime(df['unix'], unit='ms')
        df.set_index('date', inplace=True)
        init_length = len(df)
        df['sentiment_smoothed'] = df['sentiment'].rolling(int(len(df)/5)).mean()
        df.dropna(inplace=True)

        #This will be the bar chart showing how many tweets were made at every milisecond (ms)
        vol_df = df.copy()

        vol_df['volume'] = 1  #added 1 for every tweet
        ms_span = (df.index[-1] - df.index[0]).seconds * 1000 #start and end time difference times miliseconds
        rs = int(ms_span / 100)   #total delta seconds divided by 100. 

        df = df.resample('{}ms'.format(int(rs))).mean()   # take average of the sentiments for every sample of rs miliseonds
        df.dropna(inplace=True)

        vol_df = vol_df.resample('{}ms'.format(int(rs))).sum()   #calculate the sum of number of tweets for every rs miliseonds
        vol_df.dropna(inplace=True)
 
        df = df.join(vol_df['volume'])
       
        X = df.index
        Y = df.sentiment_smoothed.values
        Y2 = df.volume.values
 
        data = plotly.graph_objs.Scatter(
                x=X,
                y=Y,
                name='Sentiment',
                mode= 'lines',
                yaxis='y2',
                line = dict(color = (app_colors['sentiment-plot']),
                            width = 4,)
                )

        data2 = plotly.graph_objs.Bar(
                x=X,
                y=Y2,
                name='Volume',
                marker=dict(color=app_colors['volume-bar']),
                )


        t28 = time.time()
        print('historical app call back bitirdim:   "{}" '.format(sentiment_term), t28 - t18)

        return {'data': [data,data2],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]), # add type='category to remove gaps'
                                                          yaxis=dict(range=[min(Y2),max(Y2*4)], title='# of tweets', side='right'),
                                                          yaxis2=dict(range=[min(Y),max(Y)], side='left', overlaying='y',title='sentiment'),
                                                          title='Longer-term sentiment for: "{}"'.format(sentiment_term),
                                                          font={'color':app_colors['text']},
                                                          plot_bgcolor = app_colors['background'],
                                                          paper_bgcolor = app_colors['background'],
                                                          showlegend=False)}
        
    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')

max_size_change = .4


@app.callback(Output('sentiment-pie', 'figure'),
              [Input('input-div', 'children'),
               #Input(component_id='sentiment_term', component_property='value'),
               Input('sentiment-pie-update', 'n_intervals')]) 
def update_pie_chart(sentiment_term, n_intervals):
    t16 = time.time()
    #print('setiment pie app call back fonksiyona girdim:  ', t1)
    
    sql_term = "SELECT COUNT(tweet) AS POS FROM sentiment WHERE tweet LIKE '%"+sentiment_term+"%' AND sentiment > 0.1"
    df = pd.read_sql(sql_term, conn)
    pos= df['POS'][0]
    sql_term = "SELECT COUNT(tweet) AS NEG FROM sentiment WHERE tweet LIKE '%"+sentiment_term+"%' AND sentiment < - 0.1"
    df = pd.read_sql(sql_term, conn)
    neg= df['NEG'][0]
    labels = ['Positive tweets','Negative tweets']
    values = [pos,neg]
    colors = ['#007F25', '#800000']

    trace = go.Pie(labels=labels, values=values,
                   hoverinfo='label+percent', textinfo='value', 
                   textfont=dict(size=20, color=app_colors['text']),
                   marker=dict(colors=colors, 
                               line=dict(color=app_colors['background'], width=2)))

    t26 = time.time()
    print('setiment pie app call back bitirdim:  "{}" '.format(sentiment_term), t26 - t16)

    return {"data":[trace],'layout' : go.Layout(
                                                  title='Positive vs Negative sentiment for "{}" (longer-term)'.format(sentiment_term),
                                                  font={'color':app_colors['text']},
                                                  plot_bgcolor = app_colors['background'],
                                                  paper_bgcolor = app_colors['background'],
                                                  showlegend=True)}
   
            
external_css = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]
for css in external_css:
    app.css.append_css({"external_url": css})


if __name__ == '__main__':
    app.run_server(debug=True)
