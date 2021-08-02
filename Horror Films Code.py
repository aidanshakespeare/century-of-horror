# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 18:45:23 2021

@author: 13369
"""
#import third party libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import the dataset to pandas
raw_data = pd.read_csv("Horror Films 1927-1999.csv")

#analyze for yearly rankings
year_rank = raw_data.drop(columns = ["TITLE","SUBGENRE","DIRECTOR",
                                     "PLOT SUMMARY"])
year_rank = year_rank.groupby("YEAR")
mean_df = year_rank.mean()
mean_df = mean_df.rename(columns={"RANKING":"MEAN"})
top_df = year_rank.min()
top_df = top_df.rename(columns={"RANKING":"TOP"})
bottom_df = year_rank.max()
bottom_df = bottom_df.rename(columns = {"RANKING":"BOTTOM"})
year_rank_math = pd.merge(mean_df, top_df, left_index = True,
                              right_index = True)
year_rank_math = pd.merge(year_rank_math, bottom_df, left_index = True,
                          right_index = True)

#Creating the first visualizations
sns.set_theme(context="poster", style="darkgrid", palette="rocket", 
              font="High Tower Text")
overall_plot = sns.scatterplot(data=raw_data, x="YEAR", y="RANKING", 
                               hue="SUBGENRE", legend=False)
overall_plot.set(xlabel = "Year", ylabel = "Box Office Ranking",
                 title = "Box Office Rankings of Top Five Horror Movies By Year")
plt.show()
plt.savefig("ex1.pdf")
plt.clf()
top_ranked = sns.lineplot(data=year_rank_math, x="YEAR", y="TOP")
top_ranked.set(xlabel = "Year", ylabel = "Box Office Ranking",
               title = "Highest Ranked Horror Movies by Year")
plt.show()
plt.savefig("ex2.pdf")
plt.clf()
subgenre_plot=sns.violinplot(data=raw_data, x="YEAR", y="SUBGENRE", cut=0, 
                             scale="count",inner=None)
subgenre_plot.set(xlabel="Year",ylabel="Number of Films in Top Five",
                  title = "Subgenre Popularity Over Time")
plt.show()
plt.savefig("ex3.pdf")
plt.clf()

#Prep plot summaries for topic modeling
import re
from wordcloud import WordCloud
import gensim
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
import gensim.corpora as corpora
from pprint import pprint
raw_data["PLOT SUMMARY PROCESSED"] = \
    raw_data["PLOT SUMMARY"].map(lambda x: re.sub("[,'()\.!?]", "", x))
raw_data["PLOT SUMMARY PROCESSED"]=\
    raw_data["PLOT SUMMARY PROCESSED"].map(lambda x: x.lower())
raw_data["PLOT SUMMARY PROCESSED"]=\
    raw_data["PLOT SUMMARY PROCESSED"].map(lambda x: re.sub('"', "", x))
all_plots=",".join(list(raw_data["PLOT SUMMARY PROCESSED"].values))
total_wordcloud = WordCloud(background_color="black")
total_wordcloud.generate(all_plots)
total_wordcloud.to_image()
stop_words = stopwords.words("english")

#Create a corpus for processing
def word_list(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]
total_data = raw_data["PLOT SUMMARY PROCESSED"].values.tolist()
total_words = list(word_list(total_data))
total_words = remove_stopwords(total_words)
id2word = corpora.Dictionary(total_words)
texts = total_words
corpus = [id2word.doc2bow(text) for text in texts]

#Time to process
lda_model = gensim.models.LdaModel(corpus=corpus, id2word=id2word,
                                       num_topics=20, eval_every=1)
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

#Visualization time
import pyLDAvis.gensim_models
import pickle
import pyLDAvis
pyLDAvis.enable_notebook()
if 1==1:
    ldavis_prepared=pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    with open("ldavis_prepared_10", "wb") as f:
        pickle.dump(ldavis_prepared, f)
with open("ldavis_prepared_10", "rb") as f:
    ldavis_prepared=pickle.load(f)
pyLDAvis.save_html(ldavis_prepared, "ldavis_prepared_10.html")
ldavis_prepared