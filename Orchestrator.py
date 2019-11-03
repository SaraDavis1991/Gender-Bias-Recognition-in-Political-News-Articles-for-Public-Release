from DataReader import DataReader
from DataContracts import Article
import ApplicationConstants

reader = DataReader()
sources = reader.Load(ApplicationConstants.all_articles)

