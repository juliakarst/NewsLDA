from wordcloud import WordCloud
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from gensim.models import LdaModel
import pandas as pd
import re
import nltk
import spacy.cli
import gensim
import pickle
import time
from gensim.models import Phrases
from pprint import pprint

start = time.process_time()

nlp = spacy.load('de_core_news_md')
nlp.max_length = 4000000

""" load the data, drop unneccessary columns """
print("reading data")
# reading the CSV-file
# for 'all_articles' the very first row needs to be dropped when reading in the CSV-file
articles = pd.read_csv('/Users/juliakarst/PycharmProjects/NewsLDA/data/ww_all.csv')
articles = articles.drop(columns=['title', 'author', 'date_published', 'url', 'source'], axis=1)
#print(articles.head())

""" remove punctuation, filter and convert to lowercase """
print("preprocessing")
articles['art_text_processed'] = articles['article_text'].map(lambda x: re.sub("[,.!?]", "", str(x)))
articles['art_text_processed'] = articles['art_text_processed'].map(lambda x: re.sub("[0-9]*", "", str(x)))
articles['art_text_processed'] = articles['art_text_processed'].map(lambda x: re.sub("China[s]*", "", str(x)))
articles['art_text_processed'] = articles['art_text_processed'].map(lambda x: re.sub("Chinese[n]*", "", str(x)))
articles['art_text_processed'] = articles['art_text_processed'].map(lambda x: re.sub("chinesische[r|n|s|m]*", "", str(x)))
articles['art_text_processed'] = articles['art_text_processed'].map(lambda x: x.lower())
#print(articles['art_text_processed'].head())

#remove stopwords 
stop_words = nltk.corpus.stopwords.words("german")
stop_words.extend(['dpa', 'mann', 'frau', 'tagesspiegel', 'sd', 'sueddeutsche', 'wirtschaftswoche', 'informationen',
                   'menschen', 'zeit', 'gut', 'neue', 'infocom', 'erste', 'ersten', 'deutsche', 'presse', 'agentur',
                   'wurde', 'wurden', 'worde', 'worden', 'wäre','ware', 'wären','waren', 'innerhalb', 'pro','klein','meist',
                   'sagt','sagte', 'sagen', 'bleiben', 'kommen', 'geben', 'sagen', 'fragen', 'spiegel', 'online', 'funf',
                   'für','fur', 'über','uber', 'wegen', 'gerade', 'gilt', 'sehen','erstmals','hervor','zweit','beispielsweise',
                   'deren', 'denen', 'wohl', 'macht', 'laut', 'damals', 'trotz', 'derzeit', 'oft', 'elf', 'reden', 'treffen', 'los',
                   'sei','seien', 'dabei', 'außerdem', 'groß','gehen', 'beid','gut','schreiben','heißen','wolle','außerte',
                   'neue', 'tun', 'lasst', 'bleibt', 'weniger', 'erklärte', 'erklarte','deswegen','nie','sitzen', 'newsblog', 'blog',
                   'seit', 'mal', 'immer', 'stets', 'fast', 'bereits', 'schon', 'inzwischen', 'bisher', 'zuletzt', 'darauf',
                   'dafür','dafur','deshalb', 'darum', 'sowie', 'statt', 'stehen', 'berichten', 'finden', 'leben', 'zeigen',
                   'können','konnen', 'könne', 'konne', 'müssen','mussen', 'müsse', 'musse','lässt', 'gar', 'woran','erkennen','zeigen',
                   'eins', 'zwei', 'drei', 'vier','rund', 'etwa', 'wer', 'was', 'wen', 'wem','eigentlich', 'demnach', 'weiterhin',
                   'gibt','mehr', 'viel','viele','wenig', 'wenige', 'kommt', 'deutlich','finden', 'bringen','sehen','hoch',
                   'konnte', 'konnten','könnte', 'könnten', 'hätte','hatte', 'hätten','hatten', 'kaum','betonen','melden','weit','eher','zeigen','erreichen',
                   'land', 'jahr','jahre','jahren', 'monat','monaten', 'klar', 'ab', 'bekannt', 'wochen','woche','jahrige','vergangenen','wichtig',
                   'sollte', 'sollten', 'sollen', 'geht', 'neu', 'neuen', 'geben', 'schreiben', 'kommen', 'berichten','groß','gut','selten','jedenfalls','lesen','weder',
                   'ja', 'nein', 'stunden','eben','allerdings', 'obwohl', 'offenbar', 'zufolge', 'ganz', 'sogar', 'angesichts', 'davon','hinaus','daruber','ubrigens',
                   'muss', 'mussten', 'gegenuber','setzen','musste','müssten', 'müssen','mussen', 'müsse', 'musse','nehmen','wahren','heißen','kurz','lang','spater','stehen','tag',
                   'deshalb', 'darum', 'sowie','prozent','liegen','eigen','start', 'ende','beziehungsweise', 'zehn', 'funf','sechs','sieben','acht','neun','the','of',
                   'dürfte','durfte', 'stellen','brauchen','dürfe','durfe', 'dürften', 'durften','dürfen', 'durfen','darf', "u",'fahren','einfach','minuten','sobald','egal',
                   'montag', 'dienstag', 'mittwoch', 'donnerstag', 'freitag', 'samstag', 'sonntag', 'heute', 'gestern', 'morgen','stimmen','behaupten','helfen','mögen','glauben','gerne','naturlich',
                   'beim', 'weiß','diess','niemand','uberhaupt','versuchen','wissen','weitere', 'aktuell','jedoch', 'teil','zudem', 'lassen', 'zunächst', 'zunachst', 'erst', 'gemacht', 'bislang',
                   'januar', 'februar', 'marz','april','mai','juni','juli','august','september','oktober','november','dezember','anfang','ende','mitte','teilen','je','bestimmen',])


#printing a wordcloud for first visualisation 
print("creating wordcloud")
long_string = ','.join(list(articles['art_text_processed'].values))
wordcloud = WordCloud(stopwords=stop_words, background_color="white", max_words=500, contour_width=3, contour_color='steelblue', width=1000, height=500)
wordcloud.generate(long_string)
image = wordcloud.to_image()
image.show()


# tokenizing
# some helper functions
def sentences_to_words(sentences):
    for sentence in sentences:
        #Convert a document into a list of tokens.
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return[[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

print("creating data_words")
data = articles.art_text_processed.values.tolist()
#print(data[0]) 
data_words = list(sentences_to_words(data))
#print(data_words[0]) 
data_words = remove_stopwords(data_words)
#filter out single letters (short words could also be removed, but not recommended for this specific use case)
data_words = [w for w in data_words if len(w) > 2] 

# lemmatizing
# helper function
def lemmatizer(input_docs):
    #Lemmatizing words
    print("Now Lemmatizing")
    lemmatized_words = []
    for word_list in input_docs:
        temp_list = []
        temp_doc = nlp(' '.join(word_list))
        for word in temp_doc:
            temp_list.append(word.lemma_)
        lemmatized_words.append(temp_list)

    return lemmatized_words


#lemmatize
data_words = lemmatizer(data_words)
#filter stopwords again
data_words=[[word for word in doc if word not in stop_words] for doc in data_words]

#add bigrams
bigrams = Phrases(data_words, min_count=20)
for i in range(len(data_words)):
    for token in bigrams[data_words[i]]:
        if '_' in token:
            data_words[i].append(token)

print(data_words)
print(data_words[0])
print(len(data_words))
with open('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/data_words_ww', 'wb') as f:
    pickle.dump(data_words, f)

print(time.process_time() - start)
