#import necessary libraries
import sys
import pandas as pd
import numpy as np
import re
import sqlite3
from sqlalchemy import create_engine

#load datasets for messages and categories
def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges data of the messages and categories.

    Parameters:
    messages_filepath (string): filepath for the messages data
    categories_filepath (string): filepath for the categories data

    Returns:
    df (dataframe): returns the merged dataframe of messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merge messages and categories based on id
    df = messages.merge(categories, on = 'id')
    return df

#clean data
def clean_data(df):
    """
    Cleans and processes the data and removing duplicates.

    Parameters:
    df (dataframe): Dataframe which has to be cleaned

    Returns:
    df (dataframe): returns the cleaned dataframe
    """
    #split dataset with categories columns only
    categories = df['categories'].str.split(pat = ';', expand=True)
    #extract the column names
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x[:-2])
    #assign columns names to the categories dataset
    categories.columns = category_colnames
    #convert category values to numbers
    for column in categories:
        #using regex to get the numeric value
        categories[column] = categories[column].apply(lambda x: re.sub(r"[a-z-_]","", x))
        #convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    #drop old categories columns
    df.drop(['categories'], axis = 1, inplace = True)
    #concat messages dataset with categories dataset
    df = pd.concat([df, categories], axis = 1)
    #drop duplicates
    df.drop_duplicates(inplace = True)
    #"related" has 3 unique values - clean the data to not falsify the classification
    #check whether in column related exists value 2 and replace it with 1
    if (df[df['related'] == 2].count()['related']) > 0:
        df.replace({'related': 2}, 1, inplace = True)
    else:
        pass
    return df


def save_data(df, database_filename):
    """
    Saves dataset to a SQLite database using the provided filepath

    Parameters:
    df (dataframe): Dataframe which has to be saved
    database_filename (string): filepath including the filename

    Returns:
    no values returned
    """
    #open sqlite engine and load the sqlite-database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_data', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
