import csv
import json

rows = []
data = {}
url_index = {} 

with open(".\\Data\\breitbart.csv", 'r') as file:
    reader = csv.reader(file)
    fields = reader.next()

    data["articles"] = []
    index = 0

    reader = sorted(reader, key=lambda row: row[0])
    for row in reader:

        article = {}
        

        url = row[3]

        if ("video" in url):
            continue

        #there's duplicate records that are pulled and need to be combined into one      
        if (url in url_index):
            content = row[7]
            original_article_index = url_index[url]
            data["articles"][original_article_index]["content"] += " " + content
        else:    
            url_index[row[3]] = index

            article["title"] = row[4]
            article["url"] = row[3]
            article["subtitle"] = row[5]
            article["author"] = row[6]
            article["content"] = row[7] 
            article["date"] = row[8]

            data["articles"].append(article)

            index += 1

with open(".\\Data\\breitbart.revised.json", 'w') as write_file:
    json.dump(data, write_file, indent=2)