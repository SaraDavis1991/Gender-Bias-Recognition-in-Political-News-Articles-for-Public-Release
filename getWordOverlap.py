def wordOverlap(male = True):
    filesToOpen = ["vocabulary/output_words_top50_tfidf_fold0.txt", "vocabulary/output_words_top50_tfidf_fold1.txt", "vocabulary/output_words_top50_tfidf_fold2.txt",
                   "vocabulary/output_words_top50_tfidf_fold3.txt", "vocabulary/output_words_top50_tfidf_fold4.txt"]
    maleWordDict = {}
    femaleWordDict = {}
    wordDict = {}

    for i, file in enumerate(filesToOpen):
        lineCounter = 1
        print(file)
        f = open(file, "r")
        for line in f:
            wordList = line.split()
            if lineCounter < 62 and lineCounter > 11:
                if wordList[0] not in maleWordDict.keys() and 'Male' not in wordList[0] and 'svm' not in wordList[0]:
                    maleWordDict[wordList[0]] = [i+1]
                elif 'Male' not in wordList[0] and 'svm' not in wordList[0]:
                    maleWordDict[wordList[0]].append(i+1)
                #print(wordList[0])
                if wordList[0] not in wordDict.keys() and 'Male' not in wordList[0] and 'svm' not in wordList[0]:
                    wordDict[wordList[0]] = 'M'
            if lineCounter < 114 and lineCounter > 63:
                if wordList[0] not in femaleWordDict.keys() and 'Female' not in wordList[0] and 'svm' not in wordList[0]:
                    femaleWordDict[wordList[0]] = [i+1]
                elif 'Female' not in wordList[0] and 'svm' not in wordList[0]:
                    femaleWordDict[wordList[0]].append(i+1)
                if wordList[0] not in wordDict.keys() and 'Female' not in wordList[0] and 'svm' not in wordList[0]:
                    wordDict[wordList[0]] = 'F'
                elif 'Female' not in wordList[0] and 'svm' not in wordList[0] and wordDict[wordList[0]] != 'F':
                    wordDict[wordList[0]] = 'B'

            lineCounter +=1
    return maleWordDict, femaleWordDict, wordDict

def printDict(dictionary, fname):
    #dictionary = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse = True)}
    dictionary = {k: v for k, v in sorted(dictionary.items(), key = lambda x: len(x[1]), reverse = True)}
    f = open(fname, "w")
    f.write("Word : [Fold List Occurrence] \n")
    for key, value in dictionary.items():
        f.write(str(key) + ":" + str(value) + "\n")

maleWordDict, femaleWordDict, wordDict = wordOverlap()
print("malefold")
printDict(maleWordDict, "vocabulary/maleFoldOverlap.txt")
print("femalefold")
printDict(femaleWordDict, "vocabulary/FemaleFoldOverlap.txt")
print("bothfold")
printDict(wordDict, "vocabulary/occGenderOccurence.txt")
