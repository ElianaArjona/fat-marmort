import re
import pandas as pd
import numpy as np
import emoji
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
import demoji
import spacy
from spacy.matcher import PhraseMatcher
from spaczz.matcher import FuzzyMatcher

# initialize demoji library
# demoji.download_codes()

# Load the language model
nlp = spacy.load("es_core_news_sm")

specifications_dict = {
        "lb": "linea blanca",
        "linea blanca":"linea blanca",
        "amoblado":"amoblado",
        "muebles":"amoblado",
        "semi":"amoblado",
        "venta":"venta",
        "alquiler":"alquiler",
        "costa del este": "costa del este",
        "cde": "costa del este", # lo confunde con stops 
        "coco del mar":"coco del mar",
        "buscar":"busqueda",
        "necesito":"busqueda",
        "oferzco":"oferta",
        "ofert":"oferta"
    # Add more variations as needed
    }

# ------------ FORMAT DATA ------------
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


# -------- CLEAN DATA --------------
def remove_emojis(df, col):

    # convert emojis to text representations and remove any remaining emojis
    df[col] = df[col].apply(lambda x: re.sub(r":[^:\s]+:", " ", demoji.replace_with_desc(x)))
    return df


def clean_messages(df, col):
    """
    Cleans a DataFrame of WhatsApp messages by performing the following steps:
    1. Remove rows that contain only the string "<Multimedia omitido>"
    2. Remove emojis and any text between colons (e.g. :smile: or :round_pushpin:)
    3. Remove any URLs (http or www) from messages
    
    Args:
        df (pandas.DataFrame): the DataFrame to be cleaned
        col (str): the name of the column containing the messages
    
    Returns:
        pandas.DataFrame: the cleaned DataFrame
    """
    
    # Step 1: Remove rows that contain only "<Multimedia omitido>"
    df = df[~(df[col] == "<Multimedia omitido>")]
    
    # Step 2: Remove emojis and any text between colons (e.g. :smile: or :round_pushpin:)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    
    def remove_emojis_and_colon_text(text):
        # Remove emojis
        text = emoji_pattern.sub(r"", text)
        
        # Remove text between colons
        text = re.sub(r":[^:\s]+:", "", text)
        
        return text
    
    df[col] = df[col].apply(remove_emojis_and_colon_text)
    
    # Step 3: Remove URLs
    df[col] = df[col].str.replace(r"http\S+", "")
    df[col] = df[col].str.replace(r"www\S+", "")
    
    return df

def remove_keywords(df, col):
    # remove strings between "<" and ">" characters
    df[col] = df[col].str.replace(r"<[^>]*>", "")

    # remove strings between ":" characters
    df[col] = df[col].str.replace(r":[^:]*:", "")
        
    return df


# ------ FILTER --------------
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

# ------- MATCHER ---------------------

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
            print(fuzz.token_set_ratio( keyword.lower(), row[column].lower()) >= threshold)
            print(row[column].lower())
            print(keyword.lower())

            if fuzz.token_set_ratio( keyword.lower(), row[column].lower()) >= threshold:
                matches.append(i)

            if highlight:
                    dataframe.at[i, column] = row[column].replace(keyword, keyword.upper())

    

    # df["matches"] = matches

    return dataframe.loc[matches].drop_duplicates()

def nlp_match(df, col, dic):
    # Initialize the PhraseMatcher with your keywords
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER", )
    patterns = [nlp(text.lower()) for text in specifications_dict]
    matcher.add("Keyword", None, *patterns)

    matches = []
    for doc in nlp.pipe(df[col], disable=["parser", "ner"]): # by line
        found_matches = []
        for _, start, end in matcher(doc, ): # by word
            keyword = doc[start:end].text.lower()
            found_matches.append(dic.get(keyword) ) # Convert to lowercase
            
        found_matches = list(set(found_matches)) # Remove duplicates

        matches.append(found_matches)
    df["matches"] = matches
    return df


def extract_apartment_specifications(df, text_column):
    specifications = []
    for text in df[text_column]:
        doc = nlp(text.lower())
        spec_dict = {}
        for token in doc:
            if token.text in specifications_dict:
                spec_dict[specifications_dict[token.text]] = None

                for child in token.children:
                    if child.dep_ == "nummod":
                        child_text = child.text.replace(",", "")

                        if '-' in child_text:
                            child_text = child_text.split('-')[0]

                        try:
                            spec_dict[specifications_dict[token.text]] = int(child_text)
                        except ValueError:
                            pass
        specifications.append(spec_dict)
    df['specifications'] = specifications
    return df


# -------- FUZZY -----------------
def nlp_fuzzy_match(df, col, dic):
    keysList = list(dic.keys())
    threshold = 70
    matches = []

    # Initialize the PhraseMatcher with your keywords
    matcher = FuzzyMatcher(nlp.vocab)
    for text in keysList:
        matcher.add("SPECS",[nlp(text.lower())])

    for data in df[col]:
        found_matches = []
        doc = nlp(data)
        for _, start, end, ratio, pattern in matcher(doc): 
            if ratio >= threshold:
                value = dic.get(pattern.lower())
                keyword = value + "_" + str(ratio)
            else:
                keyword = ''
            
            found_matches.append(keyword) # Convert to lowercase
            
        found_matches = list(set(found_matches)) # Remove duplicates

        matches.append(found_matches)
    df["matches"] = matches
    return df

# --------------- MAIN ----------------

content = read_file("sample-brokers.txt")
df = split_data(content)
df = add_date_columns(df)

# filtered_df = filter_by_user(df, "Meibilin Castellanos")
# df_e = remove_emojis(df, 'msg')
# df = remove_keywords(df_e,'msg')
# keywords = ["venta", "compra", "alquiler", "inmueble", "costa del este"]
df = clean_messages(df, 'msg')


# filtered_df =  match_keywords(df, "msg", keywords, True)
# Define your keywords
filtered_df = nlp_fuzzy_match(df, "msg", specifications_dict)




# Print the results
print(filtered_df)
# print(ratio)
filtered_df.to_csv("output-sample.csv", index=False, encoding='UTF-8')


# user_name = "Meibilin Castellanos" # Replace with the user name you want to filter by
# pattern = r'Meibilin Castellanos.*' # regex pattern for user names starting with "Joe"
# filtered_df = df[df['user'].str.contains(pattern)]


# print(filtered_df)

# result.to_csv("sample.csv", index=False, encoding='UTF-8')


# df = rawToDf("sample-brokers.txt", '12hr')
# print(df)