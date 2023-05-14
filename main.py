import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
# import emoji

def remove_mutltimedia_omited(df):
    """
    Validates and cleans a DataFrame based on certain rules:
    1. Remove rows that contain only the string "<Multimedia omitido>"
    2. Replace any occurrence of the string "<Multimedia omitido>" with an empty string
    """

    df = df[~(df['msg'] == '<Multimedia omitido>')] # remove rows with only <Multimedia omitido>
    df['msg'] = df['msg'].str.replace('<Multimedia omitido>', '') # replace <Multimedia omitido> with empty string
    return df


import re
import pandas as pd

def read_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    return content

def split_data(content):
    data = []
    current_date = None
    current_user = None
    current_msg = []

    for line in content.splitlines():
        if re.match(r"\d+/\d+/\d+, \d+:\d+ - .+: ", line):
            # Add the previous message to the data
            if current_date is not None:
                data.append([current_date, current_user, " ".join(current_msg)])
            # Parse the new message
            parts = line.split(" - ", 1)
            current_date = pd.to_datetime(parts[0], format="%d/%m/%Y, %H:%M")
            user_msg = parts[1].split(": ", 1)
            current_user = user_msg[0]
            current_msg = [user_msg[1]]
        else:
            current_msg.append(line)

    # Add the last message to the data
    if current_date is not None:
        data.append([current_date, current_user, "\n".join(current_msg)])

    return pd.DataFrame(data, columns=["date", "user", "msg"])

def add_date_columns(df):
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    return df

def remove_mutltimedia_omited(df):
    """
    Validates and cleans a DataFrame based on certain rules:
    1. Remove rows that contain only the string "<Multimedia omitido>"
    2. Replace any occurrence of the string "<Multimedia omitido>" with an empty string
    """

    df = df[~(df['msg'] == '<Multimedia omitido>')] # remove rows with only <Multimedia omitido>
    df['msg'] = df['msg'].str.replace('<Multimedia omitido>', '') # replace <Multimedia omitido> with empty string
    return df


def filter_by_user(df, username):
    """
    Filters a DataFrame by a given username and returns a new DataFrame with only the rows
    containing messages from that user.
    """
    pattern = re.escape(username) + '.*'
    return df[df['user'].str.contains(pattern)]


def filter_by_date(df, year=None, month=None, day=None):
    """
    Filters a DataFrame by a given year, month, and/or day and returns a new DataFrame with only
    the rows containing messages from that date.
    """
    mask = df["date"].dt.year == year if year is not None else True
    mask &= df["date"].dt.month == month if month is not None else True
    mask &= df["date"].dt.day == day if day is not None else True
    return df[mask]




def match_keywords(dataframe, column, keywords, threshold=80, highlight=False):
    """
    Matches a list of keywords to a column of a Pandas DataFrame using the Levenshtein distance algorithm.
    Returns a new DataFrame containing only the rows that contain a match for any of the keywords.
    
    Parameters:
        - dataframe: Pandas DataFrame
        - column: name of the column to match against
        - keywords: list of keywords to match
        - threshold: minimum similarity score to consider a match (default=80)
    
    Returns:
        - new DataFrame containing only the rows that contain a match for any of the keywords
    """
    matches = []
    for keyword in keywords:
        for i, row in dataframe.iterrows():
            if fuzz.token_set_ratio( keyword.lower(), row[column].lower()) >= threshold:
                matches.append(i)
            if highlight:
                    dataframe.at[i, column] = row[column].replace(keyword, keyword.upper())
    return dataframe.loc[matches].drop_duplicates()


content = read_file("sample-brokers.txt")
df = split_data(content)
df = add_date_columns(df)

# filtered_df = filter_by_user(df, "Meibilin Castellanos")
data = remove_mutltimedia_omited(df)
filtered_df =  match_keywords(data, "msg", ['cde'], True)



# user_name = "Meibilin Castellanos" # Replace with the user name you want to filter by
# pattern = r'Meibilin Castellanos.*' # regex pattern for user names starting with "Joe"
# filtered_df = df[df['user'].str.contains(pattern)]


# print(filtered_df)

filtered_df.to_csv("sample.csv", index=False)


# df = rawToDf("sample-brokers.txt", '12hr')
# print(df)