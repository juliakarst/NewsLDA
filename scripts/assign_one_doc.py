import pickle
import gensim
from gensim.models import LdaModel
from gensim import corpora
import gensim.corpora as corpora


#Test-Dokument: https://www.tagesspiegel.de/wirtschaft/schlimmere-folgen-als-der-ukraine-krieg-so-bedroht-chinas-covid-politik-die-weltwirtschaft/28280546.html
test_datei =open('/Users/juliakarst/PycharmProjects/NewsLDA/data/new_article','r')

document = test_datei.read()


""" Load the dictionary """
news_dictionary = gensim.corpora.Dictionary.load('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/news_dictionary')
""" Load the model """
model = LdaModel.load('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/model')

doc_bow = news_dictionary.doc2bow(document.split())
print(doc_bow)
doc_lda = model[doc_bow]

topics = model.print_topics()
for i in topics:
    print(i)
print(doc_lda)
