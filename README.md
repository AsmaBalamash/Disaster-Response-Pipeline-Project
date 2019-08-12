# Disaster Response Pipeline Project

### Summary of the Project:
The aim of this project is building a machine learning pipeline to caregorize emergence messages based on the needs communicated by the sender.

This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also display visualizations of the data.

### File Description:
This repository contains three folders (data, models, app).
- In *data* folder, you will find two csv files required to run the python code file titled by '**process_data.py**' while '**DisasterResponse.db**' is the result of running this code. This python code takes two inputs (**disaster_messages.csv**, **disaster_categories.csv**) which containing message data and message categories or labels and the output will be SQLite database containing a merged and cleaned dataset.

- In *models* folder, you will find one python code titled by '**train_classifier.py**' which takes the SQLite database produced by '**process_data.py**' as input and uses this data to train the model. The output of this code will be evaluation statements of the created model printed in the console and this model will be saved in the same folder as pickle format titled by '**classifier.pkl**'.

- In *app* folder, you will find *template* folder and '**run.py**' which are required to run and render the web app.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Screenshots:
Below are a few screenshots of the web app.

![alt text](https://github.com/AsmaBalamash/Disaster-Response-Pipeline-Project/blob/master/Screenshot1.PNG)
![alt text](https://github.com/AsmaBalamash/Disaster-Response-Pipeline-Project/blob/master/Screenshot2.PNG)


