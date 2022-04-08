def wordOverlap(male = True):
    filesToOpen = ["vocabulary/output_words_top50_tfidfNN_fold0.txt", "vocabulary/output_words_top50_tfidfNN_fold1.txt", "vocabulary/output_words_top50_tfidfNN_fold2.txt",
                   "vocabulary/output_words_top50_tfidfNN_fold3.txt", "vocabulary/output_words_top50_tfidfNN_fold4.txt"]
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
                elif 'Feale' not in wordList[0] and 'svm' not in wordList[0]:
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

def svm_NN_compare():
    female_list = ["vocabulary/FemaleFoldOverlap.txt", "vocabulary/FemaleFoldOverlapNN.txt"]
    male_list = ["vocabulary/maleFoldOverlap.txt", "vocabulary/maleFoldOverlapNN.txt"]

    maleWordDict= {}
    femaleWordDict = {}

    for i, file in enumerate(female_list):
        lineCounter = 1
        f = open(file, "r")
        for line in f:
            if lineCounter != 1:
                wordsAndFolds = line.split(':')
                word = wordsAndFolds[0]
                if word not in femaleWordDict.keys() and i == 1:
                    femaleWordDict[word] = str(i) + " " + str(wordsAndFolds[1])
                    femaleWordDict[word] = femaleWordDict[word].strip("\n")
                elif word not in femaleWordDict.keys():
                    femaleWordDict[word] = str(wordsAndFolds[1])
                    femaleWordDict[word] = femaleWordDict[word].strip("\n")
                else:
                    femaleWordDict[word] += str(wordsAndFolds[1])
                    femaleWordDict[word] = femaleWordDict[word].strip("\n")
            lineCounter +=1
    for i, file in enumerate(male_list):
        lineCounter = 1
        f = open(file, "r")
        for line in f:
            if lineCounter != 1:
                wordsAndFolds = line.split(':')
                word = wordsAndFolds[0]
                if word not in maleWordDict.keys() and i == 1:
                    maleWordDict[word] = str(i) + " " + str(wordsAndFolds[1])
                    maleWordDict[word] = maleWordDict[word].strip("\n")
                elif word not in maleWordDict.keys():
                    maleWordDict[word] = str(wordsAndFolds[1])
                    maleWordDict[word] = maleWordDict[word].strip("\n")
                else:
                    maleWordDict[word] += str(wordsAndFolds[1])
                    maleWordDict[word] = maleWordDict[word].strip("\n")
            lineCounter +=1
    return maleWordDict, femaleWordDict

#maleWordDict, femaleWordDict, wordDict = wordOverlap()
#print("female")
#printDict(maleWordDict, "vocabulary/maleFoldOverlapNN.txt")
#print("male")
#printDict(femaleWordDict, "vocabulary/FemaleFoldOverlapNN.txt")
#print("bothfold")
#printDict(wordDict, "vocabulary/occGenderOccurenceNN.txt")

maleWordDict, femaleWordDict = svm_NN_compare()
printDict(maleWordDict, "vocabulary/NN_SVM_Male_Overlap.txt")
printDict(femaleWordDict, "vocabulary/NN_SVM_Female_Overlap.txt")
