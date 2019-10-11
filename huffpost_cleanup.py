
import json 

def open(filename, permission_type):
    data = {}
    with open(filename, permission_type) as outfile:
        json.dump(data, outfile)
    return data

data = open()