#################################
# DataContracts.py:
# This file provides the data definition for articles, labels, and sources. Used to transform json into object instances.
#################################

from typing import List

class Article():
    ''' This provides a definition for json articles '''

    def __init__(self, title, url, subtitle, author, content, date, label):
        self.Title = title 
        self.Url = url
        self.Subtitle = subtitle
        self.Author = author
        self.Content = content 
        self.Date = date
        self.Label : Label = label

class Label():
    ''' This provides a definition for json labels '''

    def __init__(self, author_gender, target_gender, target_affiliation, target_name):
        self.AuthorGender = author_gender
        self.TargetGender = target_gender
        self.TargetAffiliation = target_affiliation
        self.TargetName = target_name

class Source():
    ''' This provides a definiton for the json sources '''

    def __init__(self, articles):
        self.Articles : List[Article] = articles
