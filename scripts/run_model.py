import gensim.corpora as corpora
from gensim.models import LdaModel
import pickle
from pprint import pprint
from gensim.models.coherencemodel import CoherenceModel
import logging
import nltk

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

""" Load the words for the dictionary """
with open('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/data_words', 'rb') as f:
    data_words = pickle.load(f)

try:
    #load the dictionary
    id2word = corpora.Dictionary.load('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/news_dictionary')
except:
    #Create and save the dictionary
    id2word = corpora.Dictionary(data_words)


id2word.filter_extremes(no_below=5, no_above=0.5)
id2word.save('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/news_dictionary')
#Create and save the corpus
corpus = [id2word.doc2bow(text) for text in data_words]
#print(corpus[:1][0][:30])
with open('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/news_corpus', 'wb') as f:
    pickle.dump(corpus, f)

print('Number of unique tokens: %d' % len(id2word))
print('Number of documents: %d' % len(corpus))

#Decide the model parameters
num_topics = 20 # Number of topics
chunk_size = 2000  # Numbers of documents fed into the training algorithm
passes = 20  # Number of times trained on the entire corpus
iterations = 750 # Number of loops over each document
eval_every = None  # Evaluate model perplexity

lda_model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunk_size,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every,
    random_state=0

)

#Save the model
model_file = '/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/model'
lda_model.save(model_file)


#Print the topics and coherence
top_topics = lda_model.top_topics(corpus, topn=10) #top topics according to UMass Coherence
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)
pprint(top_topics)
#pprint(lda_model.print_topics())



