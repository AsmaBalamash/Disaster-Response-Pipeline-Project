import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Input: Two datasets with csv extensions.
    Output: Merged dataset.
    '''
    # Load messages.csv into a dataframe
    messages = pd.read_csv(messages_filepath)
    # Load categories.csv into a dataframe
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories,on='id') 
    return df


def clean_data(df):
    '''
    This function cleans the data by doing the following:
    1) Split categories into separate category columns.
    2) Convert category values to just numbers 0 or 1.
    3) Replace categories column in df with new category columns.
    4) Remove duplicates.
    
    Input: original dataframe
    Output: cleaned dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x.split('-')[0])
    # rename the columns of `categories`
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1.
    categories = categories.applymap(lambda x: int(x.split('-')[1]))
    # drop the original categories column from `df`  
    df = df.drop(columns='categories')
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    '''
    Save the clean dataset into an sqlite database. 
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('MessagesTable', engine, index=False, if_exists='replace')

    
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
