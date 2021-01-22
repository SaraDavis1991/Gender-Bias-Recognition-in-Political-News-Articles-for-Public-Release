from sklearn.feature_extraction.text import TfidfVectorizer
from Orchestrator import Orchestrator
import ApplicationConstants
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd

def convert_label(labels):
    lbls = []
    for label in labels:
        if label == 0:
            label = -1
        else:
            label = label
        lbls.append(label)
    labels = lbls
    return labels

'''
if svm =True, function trains and SVM to try to distinguish samples, else it uses a perceptron that holds a random 20% of the train as validation
if all = True, all candidates are mixed together and 80% is used as train 20% as test, else 2 candidates are held as test and 8 are held as train
'''
def run_tfidf(articles , i, svm = True, all = True):
    list_articles_list_train = []
    list_articles_list_val = []
    list_articles_list_test = []
    list_labels_train = []
    list_labels_test = []
    list_labels_val = []
    for j, leaning in enumerate(articles[i]):
        training_dataset = articles[i][leaning][ApplicationConstants.Train]  # load all train for fold
        validation_dataset = articles[i][leaning][ApplicationConstants.Validation]  # load all val for fold
        test_dataset = articles[i][leaning][ApplicationConstants.Test]  # load all test for fold

        train_articles = list(map(lambda article: article.Content, training_dataset))
        test_articles = list(map(lambda article: article.Content, test_dataset))
        validation_articles = list(map(lambda article: article.Content, validation_dataset))

        # append the articles for the leaning to a master list
        list_articles_list_train.append(train_articles)
        list_articles_list_val.append(validation_articles)
        list_articles_list_test.append(test_articles)

        train_labels = list(map(lambda article: article.Label.TargetGender, training_dataset))
        test_labels = list(map(lambda article: article.Label.TargetGender, test_dataset))
        validation_labels = list(map(lambda article: article.Label.TargetGender, validation_dataset))

        # append the labels for the leaning to a master list
        list_labels_train.append(train_labels)
        list_labels_test.append(test_labels)
        list_labels_val.append(validation_labels)


    # convert 2d list into 1d
    train_articles = [j for sub in list_articles_list_train for j in sub]

    validation_articles = [j for sub in list_articles_list_val for j in sub]
    test_articles = [j for sub in list_articles_list_test for j in sub]
    train_labels = [j for sub in list_labels_train for j in sub]
    train_labels = convert_label(train_labels)

    validation_labels = [j for sub in list_labels_val for j in sub]
    validation_labels = convert_label(validation_labels)
    test_labels = [j for sub in list_labels_test for j in sub]
    test_labels = convert_label(test_labels)

    if all:
        articles = train_articles + validation_articles + test_articles
        labels = train_labels + validation_labels + test_labels
        articles, labels = zip(*sorted(zip(articles, labels)))
        train_articles = articles[:int(len(articles) *.4)]
        train_labels = labels[:int(len(labels) * .4)]
        validation_articles = articles[int(len(articles) *.4):int(len(articles) *.8)]
        validation_labels = labels[int(len(labels) * .4):int(len(labels) * .8)]
        test_articles = articles[int(len(articles) *.8):]
        test_labels = labels[int(len(labels) * .8):]

    tfidf_transformer = TfidfVectorizer(use_idf =False) #can add params here
    tfidf_transformer.fit(train_articles+validation_articles+test_articles)
    train_tfidf = tfidf_transformer.transform(train_articles +validation_articles)
    test_tfidf = tfidf_transformer.transform(test_articles)

    if svm:
        from Models.SVM_engine import SVM
        net = SVM()
        net.Train(train_tfidf, train_labels + validation_labels, train_tfidf, train_labels + validation_labels)
        weights = net.Get_Weights()
        weights = weights.todense()
    else:
        from Models.NN_engine import Linear_NN
        net = Linear_NN()
        weights = net.Train(train_tfidf, train_labels + validation_labels, train_tfidf, train_labels + validation_labels)

    #net.Train(train_tfidf, train_labels + validation_labels, train_tfidf, train_labels + validation_labels)
    predictions = net.Predict(test_tfidf)  # pred on test counts
    acc = accuracy_score(test_labels, predictions)  # get accuracy
    target_names = ['Female', 'Male']
    class_rep = classification_report(test_labels, predictions, target_names=target_names)
    print("accuracy is: " + str(acc))

    print(class_rep)
    print("WEIGHTS")
    #weights = net.Get_Weights()
    #weights = weights.todense()
    df_weights = pd.DataFrame(weights.T, index=tfidf_transformer.get_feature_names(), columns=["svm_weights"])
    df_weights = df_weights.sort_values(by=["svm_weights"], ascending=False)
    topMale = (df_weights.head(50))
    topFemale = (df_weights.tail(50)).iloc[::-1]
    print("TOP MALE WORDS: ")
    print(topMale)
    print("\n")
    print("TOP FEMALE WORDS")
    print(topFemale)
    if svm:
        foutval = "vocabulary/output_words_top50_tfidf_fold" + str(i) + ".txt"
    else:
        foutval = "vocabulary/output_words_top50_tfidfNN_fold" + str(i) + ".txt"
    if all:
        foutval = "vocabulary/output_words_top50_tfidfNN_ALL.txt"
    fout = open(foutval, 'w')
    fout.write(class_rep)
    fout.write("\n")
    fout.write("Male Top Words: \n")
    fout.write(str(topMale))
    fout.write("\n")
    fout.write("Female Top Words: \n")
    fout.write(str(topFemale))
    fout.write("\n")

orchestrator = Orchestrator()
articles = orchestrator.read_data(path=ApplicationConstants.all_articles_random_v4_cleaned, number_of_articles=50
                            , save=False, random = True)
del orchestrator
#if running in loop, all should be false because you want to grab a chunk of candidates at a time

for i in range(5):
    run_tfidf(articles, i, svm = False, all =False)


#combine all candidates and do a shuffle to randomize them, then test on mix of all candidates articles (0 does not matter)
run_tfidf(articles, 0, svm = False, all = True)