import csv
import json
import os.path
from os import path

data = {}
url_index = {} 
json_already_added = []
input_file_path = ".\\Data\\breitbart.csv"
output_file_path = '.\\Data\\breitbart.revised.json'

def create_json(file, json_file):
    ''' Creates the json representation of the data
    '''
    data["articles"] = []
    index = 0

    #open .csv for reading        
    reader = csv.reader(file)

    #sort the rows by the scraper id to have the data in order   
    reader = sorted(reader, key=lambda row: row[0])

    #check for existing file, don't replace already labeled data 
    if (json_file is not None):
        revised_data = json.load(json_file)

    for row in reader:
        article = {}
        url = row[3]

        #remove any artifacts that came from video links (so pages had both videos and articles linked on the same page)
        if ("video" in url):
            continue

        #search through the already labeled data and continue if it already exists as to not overwrite the data
        if (json_file is not None):
            result = list(filter(lambda x: url == x['url'], revised_data["articles"]))
            if (any(result)):
                if (result[0]["url"] in json_already_added):
                    continue 
                else:
                    data["articles"].append(result[0])
                    json_already_added.append(result[0]["url"])
                    index += 1
                    continue

        #there's duplicate records that are pulled and need to be combined into one      
        if (url in url_index):
            content = row[7]
            original_article_index = url_index[url]
            data["articles"][original_article_index]["content"] += " " + content
        else:    
            #setup data structures
            url_index[row[3]] = index

            article["title"] = row[4]
            article["url"] = row[3]
            article["subtitle"] = row[5]
            article["author"] = row[6]
            article["content"] = row[7] 
            article["date"] = row[8]
            article["labels"] = {                  
                "author_gender" : "",
                "target_gender" : "",
                "target_affiliation" : "", 
                "target_name" : ""
            }

            data["articles"].append(article)

            index += 1

#open .csv and .json if it has already been created
#using latin1 since I originally created the file in python 2 
with open(input_file_path, 'r', encoding="Latin-1") as file:   
    if (path.exists(output_file_path)):
        with open(output_file_path, 'r') as json_file:
            create_json(file, json_file)
    else:
        create_json(file, None)
        
#write out the result
with open(output_file_path, 'w') as write_file:
    json.dump(data, write_file, indent=2)