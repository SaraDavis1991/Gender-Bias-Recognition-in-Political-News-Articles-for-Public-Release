import numpy as np
import string
import spacy




def doc_clean(document, lowercase = True, noPunct = True):
    if noPunct:
        exclude = set(string.punctuation)
        document = ''.join(ch for ch in document if ch not in exclude)
    if lowercase:
        document = document.lower()
    return document

#input: a document
#output: a list containing all ngrams in the document
def doc_word_ngram(n, document, lowercase = True, noPunct = True, pos = False):
    if not pos:
        document = doc_clean(document, lowercase, noPunct)

    words = document.split(" ")
    docNgrams = []
    for i, word in enumerate(words):
        if i + n <= len(words):
            wordString = ''
            for j in range(n):
                wordString +=words[i +j]
                wordString += ' '
            docNgrams.append(wordString)
    return np.asarray(docNgrams)



def doc_char_ngram(n, document, lowercase = True, noPunct = True):
    document = doc_clean(document, lowercase, noPunct)

    wordNgrams = []
    for i, char in enumerate(document):
        if i + n <= len(document):
            charString = ''
            for j in range(n):
                charString +=document[i+j]
            wordNgrams.append(charString)
    return np.asarray(wordNgrams)

#https://spacy.io/api/annotation
def doc_pos_ngram(n, document, lowercase = True, noPunct = True):
    document = doc_clean(document, lowercase, noPunct)
    nlp = spacy.load("en_core_web_lg")
    document = nlp(document)


    posNgrams = []
    for i, token in enumerate(document):
        if i + n <= len(document):
            wordString = ''
            for j in range(n):
                wordString += document[i +j].pos_
                wordString += ' '
            posNgrams.append(wordString)
    return posNgrams


#doc_word_ngram(2, test_doc)
#word3gram = doc_word_ngram(3, test_doc, False, False)
#doc_word_ngram(4, test_doc)
#doc_word_ngram(5, test_doc)

#doc_char_ngram(4, test_doc)
#doc_pos_ngram(3, test_doc)