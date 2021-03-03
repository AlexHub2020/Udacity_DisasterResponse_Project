# Udacity_DisasterResponse_Project
Udacity Project - ETL, NLP - Predicting Categories of Messages

1. Installation
Necessary packages to run the web app and the python files:
Python
Pandas
Numpy
NLTK
Flask
Plotly
SQLite3
SQLalchemy
SKLEARN
Pickle

2. Project Motivation
The open soure dataset of Figure Eight "Multilingual Disaster Response Messages" contains real live messages sent during disaster events. These messages and the classification to a disaster category is used to create a machine learning model. Based on that model any message can be classified to a certain disaster category by using a web flask app and identify what kind of disaster is happening and what kind of help might be needed. 

3. File Descriptions
Data source:
The data used can be found on the Figure Eight / Appen webseite within the Open Source Datasets. The following dataset was chosen "Multilingual Disaster Response Messages". 
Link: https://appen.com/datasets/combined-disaster-response-data/

Project Files within this repository:
- folder "app": The main parts for the Flask app were provided by the Udacity to complete the project. 
  - folder "templates": 
    - file "go.html": HTML page to show the result of the typed message and displays the values for the 36 disaster categories. 
    - file "master.html": Main HTML page to show the input field to type in the message as well as the visualizations / plots showing details / insights of the training data.  
  - file "run.py": Main file to start the flask app. Contains the data processing and the layout of the plots.  
- folder "data": Within the folder "data" not only the raw data is stored but also the python file to clean and prepare the data for the machine learning process.
  - file "disaster_categories.csv": Contains the values for the 36 disaster categories the messages are classified.
  - file "disaster_messages.csv": Contains the messages in English and its original language as well as the assigned genre of the message. 
  - file "process_data.py": Contains the ETL-Pipeline to load the data, clean it, prepare it for further processing and store it in a SQLite database. 
- folder "models": 
  - file "train_classifier.py": Contains the Machine Learning Processing / Pipeline - loading the SQLite database, creating a ML model by using pipeline and GridSearch, training and testing the model and storing the model as a pickle file. 

4. How To Interact With My Project
The app can be used to classify any messages with regard to 36 categories on disaster events. Furthermore the app visualize the training data on disaster messages with two specific plots. 

5.Licensing, Authors, Acknowledgements
The basic structure of the flask app (esp. folder templates) was provided by Udacity to create the app and to complete the project.
Data do disaster response messages were provided by Figure Eight: https://appen.com/datasets/combined-disaster-response-data/


6. References, Sources, Links
The following sources are used to create and develop this app and python files: 
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
https://docs.python.org/3/library/re.html
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.rstrip.html?highlight=strip
https://towardsdatascience.com/how-to-show-all-columns-rows-of-a-pandas-dataframe-c49d4507fcf 
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html?highlight=apply#pandas.DataFrame.apply
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html?highlight=merge#pandas.DataFrame.merge
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html?highlight=drop#pandas.DataFrame.drop
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_numeric.html
https://www.w3schools.com/python/python_ref_string.asp
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html#pandas.concat
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.duplicated.html?highlight=duplicate
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql_table.html
https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html?highlight=train%20test%20split#sklearn.model_selection.train_test_split
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
https://knowledge.udacity.com/questions/422261
https://knowledge.udacity.com/questions/102104
https://scikit-learn.org/dev/modules/generated/sklearn.ensemble.AdaBoostClassifier.html?highlight=adaboost#sklearn.ensemble.AdaBoostClassifier
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
https://plotly.com/python/bar-charts/#facetted-subplots
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.index.html
