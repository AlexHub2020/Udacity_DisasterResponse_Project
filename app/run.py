#import necessary libraries 
import json
import plotly
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

#provided function to transform text input of app user
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load SQlite database
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_data', engine)

# load model for disaster prediction
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    #Data for 1st visual
    #Get the Top Ten Categories of the categorized messages

    #count the values for each category and sort the values in descending order
    #store the data in a dataframe and reset the index
    cat_val =pd.DataFrame(df.iloc[:,4:].apply(np.sum, axis=0).sort_values(ascending = False).reset_index())

    #assign new names to the columns
    cat_val.columns = ['category', 'number']

    #get top 10 categories
    cat_val_10 = cat_val.iloc[:9]

    #Data for 2nd visual
    #Divide the top ten categories by genre
    #group the messages by genre
    cat_by_genre = df.groupby('genre').sum().iloc[:,1:]

    #get list of genres and sorted top ten categories
    genres = cat_by_genre.index
    categories = cat_val_10.category

    # create visuals
    graphs = [
        #1st visual to display the top 10 Categories of the classified messages
        {   #data for the barchart
            'data': [
                Bar(
                    x=cat_val_10.category,
                    y=cat_val_10.number
                )
            ],
            #Layout of the barchart
            'layout': {
                'title': 'Top 10 Categories Of Classified Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        #2nd visual to display the genres of the top 10 categories
        {   #data for the barchart - one bar chart for each genre
            'data': [
                Bar(
                    x=cat_val_10.category,
                    y=cat_by_genre.loc[genres[0],categories],
                    name=genres[0]
                ),
                Bar(
                    x=cat_val_10.category,
                    y=cat_by_genre.loc[genres[1],categories],
                    name=genres[1]
                ),
                Bar(
                    x=cat_val_10.category,
                    y=cat_by_genre.loc[genres[2],categories],
                    name=genres[2]
                )
            ],
            #layout of the barchart
            'layout': {
                'title': 'Top 10 Categories Of Classified Messages Based On Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Top 10 Category"
                }
            }
        }


    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
