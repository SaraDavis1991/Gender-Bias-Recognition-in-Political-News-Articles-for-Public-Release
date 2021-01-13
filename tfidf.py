from sklearn.feature_extraction.text import TfidfVectorizer
from Orchestrator import Orchestrator
import ApplicationConstants
import numpy as np
from Models.SVM_engine import SVM
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
def run_tfidf():
    orchestrator = Orchestrator()
    articles = orchestrator.read_data(path=ApplicationConstants.all_articles_random_v4_cleaned, number_of_articles=50
                              , save=False)
    del orchestrator
    list_articles_list_train = []
    list_articles_list_val = []
    list_articles_list_test = []
    list_labels_train = []
    list_labels_test = []
    list_labels_val = []
    for j, leaning in enumerate(articles[0]):
        training_dataset = articles[0][leaning][ApplicationConstants.Train]  # load all train for fold
        validation_dataset = articles[0][leaning][ApplicationConstants.Validation]  # load all val for fold
        test_dataset = articles[0][leaning][ApplicationConstants.Test]  # load all test for fold

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

    validation_labels = [j for sub in list_labels_val for j in sub]
    test_labels = [j for sub in list_labels_test for j in sub]
    '''
    train_articles = np.asarray(train_articles)
    print(train_articles.shape)
    validation_articles = np.asarray(validation_articles)
    test_articles = np.asarray(test_articles)
    train_labels = np.asarray(train_labels)
    validation_labels = np.asarray(validation_labels)
    test_labels = np.asarray(test_labels)
    '''

    tfidf_transformer = TfidfVectorizer() #can add params here
    #train_tfidf = tfidf_transformer.fit_transform(train_articles)
    #test_tfidf = tfidf_transformer.fit_transform(test_articles)
    #validation_tfidf = tfidf_transformer.fit_transform(validation_articles)
    tfidf_transformer.fit(train_articles+validation_articles+test_articles)
    train_tfidf = tfidf_transformer.transform(train_articles +validation_articles)
    #validation_tfidf = tfidf_transformer.transform(validation_articles)
    test_tfidf = tfidf_transformer.transform(test_articles)
    del tfidf_transformer

    net = SVM()
    #print(train_tfidf.shape, validation_tfidf.shape)

    net.Train(train_tfidf, train_labels+validation_labels, train_tfidf, train_labels+validation_labels)
    predictions = net.Predict(test_tfidf)  # pred on test counts
    acc = accuracy_score(test_labels, predictions)  # get accuracy
    target_names = ['Female', 'Male']
    class_rep = classification_report(test_labels, predictions,
                                      target_names=target_names)
    print("accuracy is: " + str(acc))

    print(class_rep)
    '''
    weights = net.Get_Weights()
    
    weights = weights[0]
    
    resTop = sorted(range(len(weights)), key=lambda sub: weights[sub])[-25:]
    resBottom = sorted(range(len(weights)), key=lambda sub: weights[sub])[:25]

    foutval = "vocabulary/output_words_top50_tfidf.txt"
    fout = open(foutval, 'w')
    fout.write(class_rep)
    fout.write("\n")
    fout.write("Male Top Words: \n")
    words = train_tfidf.get_feature_names()
    for index in resTop:
        fout.write(words[index] + ' ' + str(float(weights[index])) + '\n')
    fout.write("Female Top Words: \n")
    for index in resBottom:
        fout.write(words[index] + ' ' + str(float(weights[index])) + '\n')
    '''

run_tfidf()