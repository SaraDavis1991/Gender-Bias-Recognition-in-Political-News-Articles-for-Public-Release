from DataReader import DataReader
from DataContracts import Article
import ApplicationConstants

reader = DataReader()
sarah_data = reader.Load(ApplicationConstants.sarah)
