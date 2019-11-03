from DataReader import DataReader
from DataContracts import Article
import ApplicationConstants

reader = DataReader()
sources = reader.Load(ApplicationConstants.all_articles)

#need to pull nyt when fixed
articles = sources[ApplicationConstants.Fox] + sources[ApplicationConstants.usa_today] + sources[ApplicationConstants.Breitbart] + sources[ApplicationConstants.HuffPost]