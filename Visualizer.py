from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np 

cmap = ['red','blue']

class Visualizer():

    def hover(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            cont, ind = self.sc.contains(event)
            if cont:
                self.update_annot(ind)
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.fig.canvas.draw_idle()

    def update_annot(self, ind):

        pos = self.sc.get_offsets()[ind["ind"][0]]
        self.annot.xy = pos
        text = "{}".format(" ".join([self.articles[n].Title for n in ind["ind"]]))
        self.annot.set_text(text)
       # self.annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        self.annot.get_bbox_patch().set_alpha(0.4) 

    def plot_TSNE(self, leaning, weights, true_labels, articles):
        
        self.articles = articles

        self.genders = list(map(lambda label: 'Male' if label == 1 else 'Female', true_labels))
        tsne = TSNE(verbose=1)
        results = tsne.fit_transform(weights)

        self.fig, self.ax = plt.subplots()

        self.annot = self.ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)

        self.sc = plt.scatter(x=results[:,0], y=results[0:,1], c=true_labels, cmap=matplotlib.colors.ListedColormap(cmap))
        #self.sc = sns.scatterplot(x=results[:,0], y=results[0:,1], palette=sns.color_palette("hls", 2), hue=genders)

        #plt.setp(ax.get_legend().get_texts(), fontsize='40')
        plt.legend(loc='best', prop={'size': 20})
        plt.title('t-SNE Article Distribution for ' + leaning, fontsize=40)
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        plt.show()


