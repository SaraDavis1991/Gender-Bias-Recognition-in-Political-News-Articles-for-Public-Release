from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import numpy as np 

class Visualizer(): 

    def plot_TSNE(self, weights, true_labels):

        tsne = TSNE(verbose=1)
        results = tsne.fit_transform(weights)

        plt.figure(figsize=(16, 10))
        sns.scatterplot(x=results[:,0], y=results[0:,1], palette=sns.color_palette("hls", 2), hue=true_labels)
        plt.show()