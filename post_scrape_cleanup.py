import csv
import json

rows = []
data = {}
url_index = {} 
json_already_added = []

with open(".\\Data\\newyorktimes.csv", 'r', encoding="Latin-1") as file:   
    with open('.\\Data\\newyorktimes.revised.json', 'r') as json_file:

        revised_data = json.load(json_file)
        
        reader = csv.reader(file)
        fields = next(reader)

        data["articles"] = []
        index = 0

        reader = sorted(reader, key=lambda row: row[0])
        for row in reader:

            article = {}
            url = row[3]

            #evaulate pre-stop conditions 
            if ("video" in url):
                continue

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
                url_index[row[3]] = index

                article["title"] = row[4]
                article["url"] = row[3]
                article["subtitle"] = row[5]
                article["author"] = row[6]
                article["content"] = row[7] 
                article["date"] = row[8]

                data["articles"].append(article)

                index += 1

with open(".\\Data\\newyorktimes.revised.json", 'w') as write_file:
    json.dump(data, write_file, indent=2)