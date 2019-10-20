import os 
import json
from typing import List

json_extension = ".revised.json"
base_directory = '.\\Data\\candidate\\'
output_file_path = '.\\Data\\articles.json'
combined_data = {}

def find_files(directory_name, extension) -> List[str]:
    files_match = []

    for root, dirs, files in os.walk(directory_name):
        for file in files:
            if file.endswith(extension):
                files_match.append(file)
    return files_match

_, article_directories, _ = os.walk(base_directory).__next__()

#for each parent directory 
for article_directory in article_directories:

    #get all json children
    children = find_files(base_directory + article_directory, json_extension)

    #combine the children into one file
    for child in children:

        with open(base_directory + article_directory + '\\' + child, 'r') as read_file:
            
            data = json.load(read_file) 

            if article_directory not in combined_data:
                combined_data[article_directory] = data 
            else:           
                combined_data[article_directory]['articles'].extend(data['articles'])

    #output the combined json
with open(output_file_path, 'w') as output_file:

    json.dump(combined_data, output_file, indent=3)