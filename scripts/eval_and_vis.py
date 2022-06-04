import pickle
import gensim
import pyLDAvis.gensim
import pyLDAvis
from gensim.models import LdaModel
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from pprint import pprint
from gensim.models.coherencemodel import CoherenceModel
import seaborn as sns
import re
import pandas as pd
import spacy.cli
import csv

""" Load data_words, corpus, dictionary and model"""
with open('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/data_words', 'rb') as f:
    data_words = pickle.load(f)
with open('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/news_corpus', 'rb') as input_file:
    corpus = pickle.load(input_file)
news_dictionary = gensim.corpora.Dictionary.load('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/news_dictionary')
model = LdaModel.load('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/model')
###
with open('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/data_words_ww', 'rb') as f:
    data_words_ww = pickle.load(f)
with open('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/news_corpus_ww', 'rb') as input_file:
    corpus_ww = pickle.load(input_file)
news_dictionary_ww = gensim.corpora.Dictionary.load('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/news_dictionary_ww')
model_ww = LdaModel.load('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/model_ww')
###
with open('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/data_words_sd', 'rb') as f:
    data_words_sd = pickle.load(f)
with open('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/news_corpus_sd', 'rb') as input_file:
    corpus_sd = pickle.load(input_file)
news_dictionary_sd = gensim.corpora.Dictionary.load('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/news_dictionary_sd')
model_sd = LdaModel.load('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/model_sd')
###
with open('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/data_words_ts', 'rb') as f:
    data_words_ts = pickle.load(f)
with open('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/news_corpus_ts', 'rb') as input_file:
    corpus_ts = pickle.load(input_file)
news_dictionary_ts = gensim.corpora.Dictionary.load('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/news_dictionary_ts')
model_ts = LdaModel.load('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/model_ts')

""" Visualization """

def pyldavis(model, corpus, dictionary):
    """ opens interactive pyLDAvis visualization in browser"""
    lda_visualization = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    pyLDAvis.show(lda_visualization)


def print_wordcloud(model, rows, columns,n):
    """ plots wordclouds with
        model: the LDA model
        rows, columns: number of rows and columns for the wordclouds
        n= number of topics
    """
    topic_colors = ['red','orangered','darkgreen','mediumvioletred','blue','olive','mediumseagreen',
            'slateblue','darkviolet','deeppink','red','orangered','darkgreen','mediumvioletred','blue','olive','mediumseagreen',
            'slateblue','darkviolet','deeppink','red','orangered','darkgreen','mediumvioletred','blue','olive','mediumseagreen',
            'slateblue','darkviolet','deeppink']
    cloud = WordCloud(background_color='white',width=4000,height=2000, max_words=10,
                      color_func=lambda *args, **kwargs: topic_colors[i])
    topics = model.show_topics(num_topics=n, formatted=False)
    fig, axes = plt.subplots(rows, columns, figsize=(8,6))
    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=500)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(fontsize=12))
        plt.gca().axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()

def show_doc_attribution(model, corpus, data_words, i):
    """ prints topics, topic attribution and tokens
        model: the LDA model
        corpus: the model's corpus
        data_words: the lemmatized tokens for each document
        i: document ID
    """
    pprint(model.print_topics()) #print topics with id
    #topics = model.print_topics()
    get_doc_topics = model.get_document_topics(corpus[i])
    print("Tokens: " + str(data_words[i]))
    print("Topic Attribution for Document "+str(i+1)+" from the corpus: " + str(get_doc_topics))


""" Evaluation"""

def evaluate_cv(model,texts,dictionary):
    """ evaluates CV Coherence"""
    if __name__ == '__main__':
        cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        print("c_v coherence:" + str(cm.get_coherence()))

def evaluate_cuci(model,texts,dictionary):
    """ evaluates UCI Coherence"""
    if __name__ == '__main__':
        cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_uci')
        print("c_uci coherence:" + str(cm.get_coherence()))

def evaluate_cnpmi(model,texts,dictionary):
    """ evaluates NPMI Coherence"""
    if __name__ == '__main__':
        cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_npmi')
        print("c_npmi coherence:" + str(cm.get_coherence()))

def evaluate_cumass(model,corpus,dictionary):
    """ evaluates UMass Coherence"""
    cm = CoherenceModel(model=model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    print("c_umass coherence:" + str(cm.get_coherence()))

def coherence_per_topic(model, texts, dictionary, coh):
    """ computes the coherence for each topic
        model: the LDA model
        texts: the list of lemmatized tokens (data_words) for the model
        dictionary: the dictionary for the model
        coh: coherence "c_uci", "c_npmi", "c_v" or "u_mass"
    """
    if __name__ == '__main__':
        cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence=coh)
        topic_coherence = cm.get_coherence_per_topic()
        if topic_coherence is not None:
            return topic_coherence

""" Visualizing the Coherence"""

def convert_topics(model, top):
    """ takes only the words from the topics, discards the percentages"""
    topics = model.print_topics(num_words=top)
    topicwords = []
    for t in topics:
        l = t[1].split("+")
        new = []
        for e in l:
            e = re.sub("[0-9]*", "", e)
            e = re.sub(".\*", "", e)
            e = re.sub("\"", "", e)
            new.append(e)
        topicwords.append(new)
    return(topicwords)


def plot_topic_coherence(m, d, n, coh):
    """ plot heatmap for
        m: model
        d: data_words
        n: news_dictionary
        coh: coherence "c_uci", "c_npmi", "c_v" or "u_mass"
    """
    topics = convert_topics(m, 8)
    words = [', '.join(topic) for topic in topics]
    numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    c = coherence_per_topic(m, d, n, coh)
    if c is not None:
        data = pd.DataFrame(data=zip(topics,c), columns=['Topic', str(coh)])
        data = data.set_index('Topic-Number')
        sns.heatmap(data=data, cmap="Blues", linecolor='black', square=True, yticklabels=numbers)
        plt.yticks(rotation=0)
        plt.show()





#evaluate_cv(model, data_words, news_dictionary)
#evaluate_cuci(model, data_words, news_dictionary)
#evaluate_cnpmi(model, data_words, news_dictionary)
#evaluate_cumass(model, corpus, news_dictionary)

#plot_topic_coherence(model, data_words, news_dictionary, "u_mass")

pyldavis(model,corpus,news_dictionary)
#print_wordcloud(model, 4, 5, 20)

