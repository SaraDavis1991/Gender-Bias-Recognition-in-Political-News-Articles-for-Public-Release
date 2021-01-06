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
                    JoeBiden.append(list(map(lambda art: article.Content, allarticles)))
                elif article.Label.TargetName == 'Barack_Obama':
                    BarrackObama.append(list(map(lambda art: article.Content, allarticles)))
                elif article.Label.TargetName == 'Bernie_Sanders':
                    BernieSanders.append(list(map(lambda art: article.Content, allarticles)))
                elif article.Label.TargetName == 'Donald_Trump':
                    DonaldTrump.append(list(map(lambda art: article.Content, allarticles)))
                elif article.Label.TargetName == 'Mitch_Mcconnell':
                    MitchMcconnell.append(list(map(lambda art: article.Content, allarticles)))
                elif article.Label.TargetName == 'Hillary_Clinton':
                    HillaryClinton.append(list(map(lambda art: article.Content, allarticles)))
                elif article.Label.TargetName == 'Betsy_Devos':
                    BetsyDevos.append(list(map(lambda art: article.Content, allarticles)))
                elif article.Label.TargetName == 'Elizabeth_Warren':
                    ElizabethWarren.append(list(map(lambda art: article.Content, allarticles)))
                elif article.Label.TargetName == 'Alexandria_ocasio-cortez':
                    AlexandriaOcasioCortez.append(list(map(lambda art: article.Content, allarticles)))
                elif article.Label.TargetName == 'Sarah_Palin':
                    SarahPalin.append(list(map(lambda art: article.Content, allarticles)))
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
                    JoeBiden.append(list(map(lambda art: article.Content, allarticles)))
                elif article.Label.TargetName == 'Barack_Obama':
                    BarrackObama.append(list(map(lambda art: article.Content, allarticles)))
                elif article.Label.TargetName == 'Bernie_Sanders':
                    BernieSanders.append(list(map(lambda art: article.Content, allarticles)))
                elif article.Label.TargetName == 'Donald_Trump':
                    DonaldTrump.append(list(map(lambda art: article.Content, allarticles)))
                elif article.Label.TargetName == 'Mitch_Mcconnell':
                    MitchMcconnell.append(list(map(lambda art: article.Content, allarticles)))
                elif article.Label.TargetName == 'Hillary_Clinton':
                    HillaryClinton.append(list(map(lambda art: article.Content, allarticles)))
                elif article.Label.TargetName == 'Betsy_Devos':
                    BetsyDevos.append(list(map(lambda art: article.Content, allarticles)))
                elif article.Label.TargetName == 'Elizabeth_Warren':
                    ElizabethWarren.append(list(map(lambda art: article.Content, allarticles)))
                elif article.Label.TargetName == 'Alexandria_ocasio-cortez':
                    AlexandriaOcasioCortez.append(list(map(lambda art: article.Content, allarticles)))
                elif article.Label.TargetName == 'Sarah_Palin':
                    SarahPalin.append(list(map(lambda art: article.Content, allarticles)))
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
                    male.append(list(map(lambda art: article.Content, allarticles)))
                else :
                    female.append(list(map(lambda art: article.Content, allarticles)))
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
                    male.append(list(map(lambda art: article.Content, allarticles)))
                else:
                    female.append(list(map(lambda art: article.Content, allarticles)))

            leaning_articles.append(np.asarray(male))
            leaning_articles.append(np.asarray(female))
            leaning_articles = np.asarray(leaning_articles , dtype = object)
            allarticlesdata.append(leaning_articles)
        allarticlesdata=np.asarray(allarticlesdata, dtype = object)

    return allarticlesdata




load_data(4)

