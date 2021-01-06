'''
In this set of analysis, we only consider adjectives that were in sentences containing target name

1) analyze adjectives per person (probably not interesting)
2) analyze adjectives per person per leaning
3) analyze adjectives per gender (probably not interesting)
4) analyze adjectives per gender per leaning
'''
from Orchestrator import Orchestrator
import ApplicationConstants
import numpy as np


def load_data(testNum):
    #path, savePath = None, clean = True, save = False, random = False, number_of_articles = 50, pos_tagged = False
    orchestrator = Orchestrator()
    articles = orchestrator.read_data(path=ApplicationConstants.all_articles_random_v4_cleaned_pos_candidate_names, savePath=None, save = False, clean = False, random =False, number_of_articles = 1000, pos_tagged = False)
    if testNum == 1:
        JoeBiden, BarrackObama, BernieSanders, DonaldTrump, HillaryClinton, BetsyDevos, ElizabethWarren, AlexandriaOcasioCortez,\
        SarahPalin, MitchMcconnell = [], [], [], [], [], [], [], [], [], []
        for j, leaning in enumerate(articles[0]):
            training_dataset = articles[0][leaning][ApplicationConstants.Train]  # load all train for fold
            validation_dataset = articles[0][leaning][ApplicationConstants.Validation]  # load all val for fold
            test_dataset = articles[0][leaning][ApplicationConstants.Test]  # load all test for fold
            allarticles = training_dataset + validation_dataset + test_dataset

            for article in allarticles:
                if article.Label.TargetName == 'Joe_Biden':
                    JoeBiden.append(article.Content)
                elif article.Label.TargetName == 'Barack_Obama':
                    BarrackObama.append(article.Content)
                elif article.Label.TargetName == 'Bernie_Sanders':
                    BernieSanders.append(article.Content)
                elif article.Label.TargetName == 'Donald_Trump':
                    DonaldTrump.append(article.Content)
                elif article.Label.TargetName == 'Mitch_Mcconnell':
                    MitchMcconnell.append(article.Content)
                elif article.Label.TargetName == 'Hillary_Clinton':
                    HillaryClinton.append(article.Content)
                elif article.Label.TargetName == 'Betsy_Devos':
                    BetsyDevos.append(article.Content)
                elif article.Label.TargetName == 'Elizabeth_Warren':
                    ElizabethWarren.append(article.Content)
                elif article.Label.TargetName == 'Alexandria_ocasio-cortez':
                    AlexandriaOcasioCortez.append(article.Content)
                elif article.Label.TargetName == 'Sarah_Palin':
                    SarahPalin.append(article.Content)
        allarticlesdata = []
        allarticlesdata.append(np.asarray(JoeBiden))
        allarticlesdata.append(np.asarray(BarrackObama))
        allarticlesdata.append(np.asarray(BernieSanders))
        allarticlesdata.append(np.asarray(DonaldTrump))
        allarticlesdata.append(np.asarray(MitchMcconnell))
        allarticlesdata.append(np.asarray(HillaryClinton))
        allarticlesdata.append(np.asarray(BetsyDevos))
        allarticlesdata.append(np.asarray(ElizabethWarren))
        allarticlesdata.append(np.asarray(AlexandriaOcasioCortez))
        allarticlesdata.append(np.asarray(SarahPalin))
        allarticlesdata = np.asarray(allarticlesdata)
    elif testNum == 2:
        allarticlesdata = []
        for j, leaning in enumerate(articles[0]):
            JoeBiden, BarrackObama, BernieSanders, DonaldTrump, HillaryClinton, BetsyDevos, ElizabethWarren, AlexandriaOcasioCortez, \
            SarahPalin, MitchMcconnell = [], [], [], [], [], [], [], [], [], []
            leaning_articles = []
            training_dataset = articles[0][leaning][ApplicationConstants.Train]  # load all train for fold
            validation_dataset = articles[0][leaning][ApplicationConstants.Validation]  # load all val for fold
            test_dataset = articles[0][leaning][ApplicationConstants.Test]  # load all test for fold
            allarticles = training_dataset + validation_dataset + test_dataset

            for article in allarticles:
                if article.Label.TargetName == 'Joe_Biden':
                    JoeBiden.append(article.Content)
                elif article.Label.TargetName == 'Barack_Obama':
                    BarrackObama.append(article.Content)
                elif article.Label.TargetName == 'Bernie_Sanders':
                    BernieSanders.append(article.Content)
                elif article.Label.TargetName == 'Donald_Trump':
                    DonaldTrump.append(article.Content)
                elif article.Label.TargetName == 'Mitch_Mcconnell':
                    MitchMcconnell.append(article.Content)
                elif article.Label.TargetName == 'Hillary_Clinton':
                    HillaryClinton.append(article.Content)
                elif article.Label.TargetName == 'Betsy_Devos':
                    BetsyDevos.append(article.Content)
                elif article.Label.TargetName == 'Elizabeth_Warren':
                    ElizabethWarren.append(article.Content)
                elif article.Label.TargetName == 'Alexandria_ocasio-cortez':
                    AlexandriaOcasioCortez.append(article.Content)
                elif article.Label.TargetName == 'Sarah_Palin':
                    SarahPalin.append(article.Content)
            leaning_articles.append(np.asarray(JoeBiden))
            leaning_articles.append(np.asarray(BarrackObama))
            leaning_articles.append(np.asarray(BernieSanders))
            leaning_articles.append(np.asarray(DonaldTrump))
            leaning_articles.append(np.asarray(MitchMcconnell))
            leaning_articles.append(np.asarray(HillaryClinton))
            leaning_articles.append(np.asarray(BetsyDevos))
            leaning_articles.append(np.asarray(ElizabethWarren))
            leaning_articles.append(np.asarray(AlexandriaOcasioCortez))
            leaning_articles.append(np.asarray(SarahPalin))
            leaning_articles = np.asarray(leaning_articles , dtype = object)
            allarticlesdata.append(leaning_articles)
        allarticlesdata=np.asarray(allarticlesdata, dtype = object)
        #print(allarticlesdata[0][0][0]) #first dim is the leaning second dim is the person third dim is an article

    elif testNum ==3:
        male, female = [], []
        for j, leaning in enumerate(articles[0]):
            training_dataset = articles[0][leaning][ApplicationConstants.Train]  # load all train for fold
            validation_dataset = articles[0][leaning][ApplicationConstants.Validation]  # load all val for fold
            test_dataset = articles[0][leaning][ApplicationConstants.Test]  # load all test for fold
            allarticles = training_dataset + validation_dataset + test_dataset

            for article in allarticles:
                if article.Label.TargetGender == 1:
                    male.append(article.Content)
                else :
                    female.append(article.Content)
        allarticlesdata = []
        allarticlesdata.append(np.asarray(male))
        allarticlesdata.append(np.asarray(female))
        allarticlesdata = np.asarray(allarticlesdata)
    elif testNum == 4:
        allarticlesdata = []
        for j, leaning in enumerate(articles[0]):
            male, female = [], []
            leaning_articles = []
            training_dataset = articles[0][leaning][ApplicationConstants.Train]  # load all train for fold
            validation_dataset = articles[0][leaning][ApplicationConstants.Validation]  # load all val for fold
            test_dataset = articles[0][leaning][ApplicationConstants.Test]  # load all test for fold
            allarticles = training_dataset + validation_dataset + test_dataset

            for article in allarticles:
                if article.Label.TargetGender == 1:
                    male.append(article.Content)
                else:
                    female.append(article.Content)

            leaning_articles.append(np.asarray(male))
            leaning_articles.append(np.asarray(female))
            leaning_articles = np.asarray(leaning_articles , dtype = object)
            allarticlesdata.append(leaning_articles)
        allarticlesdata=np.asarray(allarticlesdata, dtype = object)

    return allarticlesdata

def map_words(articles_corpus):
    dictionary_list = []
    uninteresting = ["gpe", "person", "people", "-"]
    for persons_articles in articles_corpus:
        word_dict = {}
        for article in persons_articles:
            print(article)
            words = article.lower().split()
            for word in words:
                if word not in word_dict and word not in uninteresting:
                    word_dict[word] = 1
                elif word in word_dict and word not in uninteresting:
                    word_dict[word] += 1
        dictionary_list.append(word_dict)
    return dictionary_list

def print_words(word_counts, file_name):
    fout = open(file_name, 'w')
    person_list = ["BIDEN", "OBAMA", "SANDERS", "TRUMP", "MCCONNELL", "CLINTON", "DEVOS", "WARREN", "CORTEZ", "PALIN"]
    for i, person in enumerate(word_counts):
        fout.write(person_list[i] + '\n')
        sorted_person = sorted(person.items(), key = lambda x: x[1], reverse = True)
        print(sorted_person[:1])
        for word, count in sorted_person[:50]:
            fout.write(word + " " + str(count) +'\n')
        fout.write('\n')

#load_data(4)
def analyze_adjectives_per_person():
    #order is biden, obama, sanders, trump, mcconnell, clinton, devos, warren cortez, palin
    articles = load_data(1)
    word_counts = map_words(articles)
    print_words(word_counts, "adjective_analysis_by_person.txt")
    #print(articles.shape)

analyze_adjectives_per_person()
