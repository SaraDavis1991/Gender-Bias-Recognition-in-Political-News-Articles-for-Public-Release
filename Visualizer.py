from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np 

class Visualizer(): 

    def plot_TSNE(self, leaning, weights, true_labels):
        
        
        genders = list(map(lambda label: 'Male' if label == 1 else 'Female', true_labels))
        tsne = TSNE(verbose=1)
        results = tsne.fit_transform(weights)
        plt.figure(figsize=(16, 10))
        #plt.scatter(x=results[:,0], y=results[0:,1], label=genders)
        scatter = sns.scatterplot(x=results[:,0], y=results[0:,1], palette=sns.color_palette("hls", 2), hue=genders)
        plt.setp(scatter.get_legend().get_texts(), fontsize='40')
        plt.legend(loc='best', prop={'size': 20})
        plt.title('t-SNE Article Distribution for ' + leaning, fontsize=40)

        plt.show()