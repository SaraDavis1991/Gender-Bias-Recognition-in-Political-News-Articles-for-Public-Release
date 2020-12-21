import os
import ngram
import numpy as np
import pickle

def get_splits():
    testFiles = []
    trainFiles = []
    trainNames = []
    testNames = []
    trainTestNamesCombos = []
    #root will give path name to each un
    #files gives a list of each file in each path
    for root, _, files in os.walk("Reddit_Cross-Topic-AV-Corpus_(1000_users)"):
        trainTestNameCombo = []
        rootFilePairTrain = []
        rootFilePairTest = []

        username = root.replace("Reddit_Cross-Topic-AV-Corpus_(1000_users)/", "")
        if username not in trainTestNameCombo:
            trainTestNameCombo.append(username)
        trainNames.append(username)
        rootFilePairTrain.append(root)
        rootFilePairTest.append(root)

        for name in files:

            if "POS_" not in name:
                if 'unknown - ' in name:
                    #print(name)
                    testNameList = name.split(' - ')
                    testName = testNameList[1]
                    testNames.append(testName)
                    trainTestNameCombo.append(testName)
                    rootFilePairTest.append(name)
                else:
                    rootFilePairTrain.append(name)
        testFiles.append(np.asarray(rootFilePairTest))
        trainFiles.append(np.asarray(rootFilePairTrain))
        trainTestNamesCombos.append(np.asarray(trainTestNameCombo))
    trainTestNamesCombos= np.asarray(trainTestNamesCombos[1:])
    testFiles = np.asarray(testFiles[1:]) #each index contains name, followed by the files associated with name (1 file for test, 3 for train)
    trainFiles = np.asarray(trainFiles[1:])
    trainNames = np.asarray(trainNames[1:])
    return (testFiles, trainFiles, trainNames, testNames, trainTestNamesCombos)

def construct_vocabulary(ngram_dict,  ngrams):
    for ngram in ngrams:
        #print(ngram)
        if ngram in ngram_dict.keys():
            ngram_dict[ngram] += 1
        else:
            ngram_dict[ngram] = 1

    #print(vocabulary)

def winnow_vocab(ngram_dict):
    #sorted_dict = dict(sorted(ngram_dict.items(), key = lambda item : item[1], reverse = True))
    #keys = list(sorted_dict.keys())[:300000]
    #vocab_counter = dict.fromkeys(keys, 0)
    vocab = {}

    for word, count in ngram_dict.items():
        if count > 1:
            vocab[word] = count
    return vocab
def build_bow(vocab_counter, article):
    #print(len(vocab))
    #vocab_counter = dict.fromkeys(vocab, 0)
    #for i in range(len(vocab)):
    #    vocab_counter.append(0)

    for word in article:
        #ind = vocab.index(word)
        #vocab_counter[ind] +=1
        if word not in vocab_counter:
            word = 'UNK'
        vocab_counter[word] +=1


def construct_counts(vocab_counter, corpus):
    counted_set = []
    keys = vocab_counter.keys()

    for i, text in enumerate(corpus):
        vocab_counter = dict.fromkeys(keys, 0)
        build_bow(vocab_counter, text)
        counted_set.append(vocab_counter)
        #print(i)
    return np.asarray(counted_set)

def run_bow(filePaths, n, charGrams = True, posGrams =True, wordGrams = True, train = True):
    #char_2gram_vocab = set()
    #char_3gram_vocab = set()

    char_gram_vocab = {}
    word_gram_vocab = {}
    pos_gram_vocab = {}
    word_gram_corpus = []
    char_gram_corpus = []
    pos_gram_corpus = []
    i = 0
    print("CONSTRUCTING VOCAB")
    print(train)
    for userList in filePaths:
        for filetoopen in userList[1:]:
            path = os.path.join(userList[0], filetoopen)
            posfile = "POS_"+str(filetoopen)
            pos_path = os.path.join(userList[0], posfile)
            fread = open(path, "r")
            freadpos = open(pos_path, "r")
            filecontents = fread.read() #contents of file
            posfilecontents = freadpos.read()
            if n == 4 and charGrams:
                #print("CONSTRUCTING CHAR VOCAB ", str(i) )
                filecontent_char_gram = ngram.doc_char_ngram(n, filecontents)
                if train:
                    construct_vocabulary(char_gram_vocab, filecontent_char_gram)
                else:
                    #print("loading train char vocab")
                    vocab_file = "./store/char_"+str(n) + "gram_vocab_train.pkl"
                    f = open(vocab_file, "rb")
                    char_gram_vocab = pickle.load(f)
                    f.close()
                char_gram_corpus.append(filecontent_char_gram)
            if posGrams:
                #print("CONSTRUCTING POS VOCAB ", str(i))
                #reading from pos file and doing wordgram with it
                filecontent_pos_gram = ngram.doc_word_ngram(n, posfilecontents, pos=True)
                if train:
                    construct_vocabulary(pos_gram_vocab, filecontent_pos_gram)
                else:
                    #print("loading train pos vocab")
                    vocab_file = "./store/pos_"+str(n) + "gram_vocab_train.pkl"
                    f = open(vocab_file, "rb")
                    pos_gram_vocab = pickle.load( f)
                    f.close()
                pos_gram_corpus.append(filecontent_pos_gram)
            if wordGrams:
                #print("CONSTRUCTING WORD VOCAB ", str(i))
                filecontent_word_gram = ngram.doc_word_ngram(n, filecontents)
                if train:
                    construct_vocabulary(word_gram_vocab, filecontent_word_gram)
                else:
                    #print("loading train word vocab")
                    vocab_file = "./store/word_"+str(n) + "gram_vocab_train.pkl"
                    f = open(vocab_file, "rb")
                    word_gram_vocab = pickle.load( f)
                    f.close()
                #print(word_gram_vocab)
                #print(word_gram_vocab)
                word_gram_corpus.append(filecontent_word_gram)
                #if i == 30:
                #    print(filecontents)
            i+=1
    pos_corpus_counts = None
    word_corpus_counts = None
    char_corpus_counts = None
    if wordGrams:
        print("CONSTRUCTING WORD COUNTS")
        print(len(word_gram_vocab))
        if len(word_gram_vocab) > 300000:
            word_vocab = winnow_vocab(word_gram_vocab)
            word_vocab['UNK'] = 1
        else:
            word_vocab = word_gram_vocab
            word_vocab['UNK'] = 1
        print(len(word_vocab))
        del word_gram_vocab
        #vocab_counter = dict.fromkeys(word_vocab, 0)
        if train:
            vocab_numpy_save = "./store/word_"+str(n) + "gram_vocab_train.pkl"
        else:
            vocab_numpy_save = "./store/word_" + str(n) + "gram_vocab_test.pkl"
        f = open(vocab_numpy_save, "wb")
        pickle.dump(word_vocab, f)
        f.close()
        word_corpus_counts = construct_counts(word_vocab, word_gram_corpus)
        del word_vocab
        del word_gram_corpus
        #del vocab_counter
    if posGrams:
        print("CONSTRUCTING POS COUNTS")
        pos_vocab = pos_gram_vocab
        pos_vocab = winnow_vocab(pos_gram_vocab)

        #print(len(pos_vocab))
        del pos_gram_vocab
        #vocab_counter = dict.fromkeys(pos_gram_vocab, 0)
        if train:
            vocab_numpy_save = "./store/pos_"+str(n) + "gram_vocab_train.pkl"
        else:
            vocab_numpy_save = "./store/pos_"+str(n) + "gram_vocab_test.pkl"
        pos_vocab['UNK'] = 1
        f = open(vocab_numpy_save, "wb")
        pickle.dump(pos_vocab, f)
        f.close()
        pos_corpus_counts = construct_counts(pos_vocab, pos_gram_corpus)
        del pos_gram_corpus
        del pos_vocab
    if charGrams:
        print("CONSTRUCTING CHAR COUNTS")
        #char_vocab = winnow_vocab(char_gram_vocab)

        #print(len(char_vocab))
        char_vocab=char_gram_vocab
        char_vocab['UNK'] = 1
        del char_gram_vocab
        #vocab_counter = dict.fromkeys(char_gram_vocab, 0)
        if train:
            vocab_numpy_save = "./store/char_"+str(n) + "gram_vocab_train.pkl"
        else:
            vocab_numpy_save = "./store/char_" + str(n) + "gram_vocab_train.pkl"
        f = open(vocab_numpy_save, "wb")
        pickle.dump(char_vocab, f)
        f.close()
        char_corpus_counts = construct_counts(char_vocab, char_gram_corpus)
        del char_gram_corpus
        del char_vocab
    return char_corpus_counts, word_corpus_counts, pos_corpus_counts

def save_corpus(corpus, n, s, train = True):

    if train:
        print("SAVING TRAIN CORPUSES")
        if s == 'char':
            corpus_numpy_save = "./store/char_" + str(n) + "gram_trainCorpus.npy"
        if s == 'pos':
            corpus_numpy_save = "./store/pos_" + str(n) + "gram_trainCorpus.npy"
        if s == 'word':
            corpus_numpy_save = "./store/word_" + str(n) + "gram_trainCorpus.npy"

    else:
        print("SAVING TEST CORPUSES")
        if s == 'char':
            corpus_numpy_save = "./store/char_" + str(n) + "gram_testCorpus.npy"
        if s == 'pos':
            corpus_numpy_save = "./store/pos_" + str(n) + "gram_testCorpus.npy"
        if s == 'word':
            corpus_numpy_save = "./store/word_" + str(n) + "gram_testCorpus.npy"
    np.save(corpus_numpy_save, corpus)

def save_names(names, s, combos = False):
    if not combos:
        if s == 'train':
            print("saving train")
            numpy_save = "./store/train_names.npy"
        if s == 'test':
            print("saving test")
            numpy_save = "./store/test_names.npy"
    else:
        print("saving combos")
        numpy_save = "./store/train_test_combos.npy"
    np.save(numpy_save, names)

def save_files(file, s):
    if s == "train":
        print("saving train")
        numpy_save = "./store/train_filenames_usernames_combos.npy"
    if s == "test":
        print("saving test")
        numpy_save = "./store/test_filenames_usernames_combos.npy"
    np.save(numpy_save, file)

def print_from_corpus(corpus_counts, person_num):
    keys = list(corpus_counts[0].keys())[:20]
    for key in keys:
        print(key,corpus_counts[person_num][key])

def run_loop(files, train = True):
    for i in range(5, 6):
        print(i)
        if i == 4:
            truth = True
        else:
            truth = False
        char_gram_corpus_counts, word_gram_corpus_counts,  pos_gram_corpus_counts\
            = run_bow(files,n = i, charGrams =truth, posGrams = True, wordGrams = False, train=train) #chargrams should be false unless n = 4
        if truth == True:
            save_corpus(char_gram_corpus_counts, i, "char", train = train)
            del char_gram_corpus_counts
        if word_gram_corpus_counts is not None:
            save_corpus(word_gram_corpus_counts, i, "word", train = train)
        del word_gram_corpus_counts
        if pos_gram_corpus_counts is not None:
            save_corpus(pos_gram_corpus_counts, i, "pos", train = train)
        del pos_gram_corpus_counts

def driver():
    if os.path.exists("./store/") == False:
        os.mkdir("./store")
    if os.path.exists("./store/train_filenames_usernames_combos.npy"):
        print("Files exist! Loading")
        trainFiles = np.load("./store/train_filenames_usernames_combos.npy")

    else:
        print("Files don't exist, creating...")
        testFiles, trainFiles, trainUsernames, testUsernames, trainTestNamesCombos = get_splits()

        trainUsernamesElongation = []
        #need to extend train names so that there are 4 to match each file in root
        for name in trainUsernames:
            trainUsernamesElongation.extend([name for i in range(4)])
        trainUsernamesElongation = np.asarray(trainUsernamesElongation)

        del trainUsernames

        save_names(trainUsernamesElongation, "train")
        save_names(testUsernames, "test")
        save_names(trainTestNamesCombos, "doesn't matter", combos = True)

        del trainUsernamesElongation
        del testUsernames
        del trainTestNamesCombos

        save_files(trainFiles, "train")
        save_files(testFiles, "test")
        del testFiles

    run_loop(trainFiles, train=True)
    print(len(trainFiles))
    del trainFiles
    testFiles = np.load("./store/test_filenames_usernames_combos.npy")
    run_loop(testFiles, train= False)
    del testFiles



    #need to uncomment 125 + 126 and change == to same num for comparison
    #print_from_corpus(word_gram_corpus_counts, 30)




driver()

#train_files = np.load("./store/train_filenames_usernames_combos.npy")
#print(train_files[0])