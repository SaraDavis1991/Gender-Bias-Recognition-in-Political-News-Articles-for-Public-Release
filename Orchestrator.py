from DataReader import DataReader
import ApplicationContants

reader = DataReader()
nyt_data = reader.Load(ApplicationContants.NewYorkTimes_DataPath)

t = 5 
nyt_data = "hi"
print(nyt_data.articles[0].author)
