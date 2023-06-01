import pandas as pd
import re
import os
from utils.utils import (
                        remove_emojis_and_colon_text, 
                        remove_special_characters,
                        remove_spanish_stopwords,
                        specifications_dict,
                        nlp
                        )
from spaczz.matcher import FuzzyMatcher

ouput_names = {
    "date_y":"year",
    "date_m":"month",
    "date_d":"day",
    "user":"contact",
    "msg":"message",
}

class DataExtractor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = None
        self.file_extension = str

    def read_file(self):
        _, self.file_extension = os.path.splitext(self.file_path)

        if self.file_extension == '.txt':
            with open(self.file_path, "r") as f:
                self.raw_data = f.read()

        if self.file_path == '.csv':
            self.raw_data = pd.read_csv(self.file_path, encoding='UTF-8')

class DataTransform:
    def __init__(self, data):
        self.raw_data = data
        self.keywords =  specifications_dict
        self.data_transform = None
    
    def split_data(self):
        data = []
        current_date, current_user, current_msg = None, None, []

        for line in self.raw_data.splitlines():
            if re.match(r"\d+/\d+/\d+, \d+:\d+ - .+: ", line):
                # Add the previous message to the data
                if current_date is not None:
                    current_date
                    data.append([current_date.year, 
                                 current_date.month,
                                 current_date.day,
                                 current_user, 
                                 " ".join(current_msg)])

                # Parse the new message
                parts = line.split(" - ", 1)
                current_date = pd.to_datetime(parts[0], format="%d/%m/%Y, %H:%M")

                # Separte data
                user_msg = parts[1].split(": ", 1)
                current_user = user_msg[0]
                current_msg = [user_msg[1]]
            else:
                # Case when a message has multiple lines
                current_msg.append(line)

            self.data_transform = pd.DataFrame(data, columns=[ ouput_names["date_y"],
                                                              ouput_names["date_m"],
                                                              ouput_names["date_d"],
                                                              ouput_names["user"],
                                                              ouput_names['msg']
                                                             ])

        # Add the last message to the data
        if current_date is not None:
            data.append([current_date, current_user, "\n".join(current_msg)])
    
    def clear_messages(self, col = ouput_names['msg'] ):
        # Step 1: Remove rows that contain only "<Multimedia omitido>"
        self.data_transform =  self.data_transform[~( self.data_transform[col] == "<Multimedia omitido>")]
        
        # Step 2: Remove Emojies from Text
        self.data_transform[col] =  self.data_transform[col].apply(remove_emojis_and_colon_text)
        
        # Step 3: Remove URLs
        self.data_transform[col] =  self.data_transform[col].str.replace(r"http\S+", "")
        self.data_transform[col] =  self.data_transform[col].str.replace(r"www\S+", "")

        # Step 4: Remove stops words
        self.data_transform[col] =  self.data_transform[col].apply(remove_spanish_stopwords)

        #Step 5: Remove accents and special charactes
        self.data_transform[col] =  self.data_transform[col].apply(remove_special_characters)

    def fuzzy_match(self, col = ouput_names['msg']):
        keysList = list(self.keywords.keys())
        threshold = 70
        matches = []

        # Initialize the PhraseMatcher with your keywords
        matcher = FuzzyMatcher(nlp.vocab)
        for text in keysList:
            matcher.add("SPECS",[nlp(text.lower())])

        for data in self.data_transform[col]:
            found_matches = []
            doc = nlp(data)
            for _, _, _, ratio, pattern in matcher(doc): 
                if ratio >= threshold:
                    value = self.keywords.get(pattern.lower())
                    keyword = value + "_" + str(ratio)
                else:
                    keyword = ''
                
                found_matches.append(keyword) # Convert to lowercase
                
            found_matches = list(set(found_matches)) # Remove duplicates

            matches.append(found_matches)

        self.data_transform["matches"] = matches

class DataLoader:
    def __init__(self, file_path, data):
        self.data = data
        self.file_path =file_path
    
    def save_file(self):
        self.data.to_csv(self.file_path, index=False, encoding='UTF-8')
    
    def print_data(self):
        print(self.data)

class DataManipulations:
    def __init__(self, filters, sort_order = None):
        self.data = None
        self.filters = filters
        self.sort_order = sort_order

    
    def apply_filters(self, data):
        # Create the regular expression pattern
        pattern = '|'.join(self.filters)

        # Filter the DataFrame based on values in 'column_name' matching the pattern
        self.data = data[
                                data["matches"].str.contains(pattern)
                                ]
    def sort_data(self, data, order):
        if order == 'asc':
            self.data = data.sort_values(by=['date', 'user'])
        else:
            self.data = data.sort_values(by=['date', 'user'])





    