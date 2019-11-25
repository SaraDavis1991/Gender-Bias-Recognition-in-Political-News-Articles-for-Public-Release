import csv
import json
import os.path
from os import path
import operator 
import ApplicationConstants 

data = {}
url_to_article_mapping = {} 
json_already_added = []
input_file_path = ".\\newData\\Data\\candidate\\breitbart\\breitbart_hillary_clinton.csv"

def create_json(file):
    ''' Creates the json representation of the data
    '''
    data["articles"] = []
    index = 0

    #open .csv for reading        
    reader = csv.reader(file)
    fields = next(reader)
    #sort the rows by the scraper id to have the data in order   
    reader = sorted(reader, key=lambda row: int(row[0].replace('-', '')))

    title_index = 4
    subtitle_index = 5 
    author_index = 6
    date_index = 7
    url_index = 3
    content_index = 8
    count = 0 
    for row in reader:
        article = {}
        url = row[url_index]
        #remove any artifacts that came from video links (so pages had both videos and articles linked on the same page)
        if ("video" in url):
            continue

        #there's duplicate records that are pulled and need to be combined into one      
        if (url in url_to_article_mapping):
            content = row[content_index]
            original_article_index = url_to_article_mapping[url]
            data["articles"][original_article_index]["content"] += " " + content
        else:    
            #setup data structures
            url_to_article_mapping[row[url_index]] = index

            article["title"] = row[title_index]
            article["url"] = row[url_index]
            article["subtitle"] = row[subtitle_index]
            article["author"] = row[author_index]
            article["content"] = row[content_index] 
            article["date"] = row[date_index]
            article["labels"] = {                  
                "author_gender" : "",
                "target_gender" : ApplicationConstants.Female,
                "target_affiliation" : ApplicationConstants.Left, 
                "target_name" : ApplicationConstants.ElizabethWarren  
            }

            data["articles"].append(article)
            count += 1
            index += 1
    return count

#open .csv and .json if it has already been created
#using latin1 since I originally created the file in python 2 
with open(input_file_path, 'r', encoding="UTF-8-sig") as file:   
    output_file_path = input_file_path.split('.')[1]
    output_file_path = ".\\" + output_file_path + '.revised.json'
    if (not path.exists(output_file_path)):
        count = create_json(file)
        print(count)    
#write out the result
with open(output_file_path, 'w') as write_file:
    json.dump(data, write_file, indent=2)