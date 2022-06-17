import pickle
import gensim
from gensim.models import LdaModel
import pprint
import csv
import re
from pprint import pprint


""" Load data_words, corpus, dictionary and model"""
with open('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/data_words', 'rb') as f:
    data_words = pickle.load(f)
with open('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/news_corpus', 'rb') as input_file:
    corpus = pickle.load(input_file)
news_dictionary = gensim.corpora.Dictionary.load('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/news_dictionary')
model = LdaModel.load('/Users/juliakarst/PycharmProjects/NewsLDA/data/modeldata/model')


def sub(text):
    """ unifying dates to a single format"""
    res = re.sub(",", "", text)
    res = re.sub(" Uhr", "", res)
    res = re.sub("Januar", "01.", res)
    res = re.sub("Februar", "02.", res)
    res = re.sub("März", "03.", res)
    res = re.sub("April", "04.", res)
    res = re.sub("Mai", "05.", res)
    res = re.sub("Juni", "06.", res)
    res = re.sub("Juli", "07.", res)
    res = re.sub("August", "08.", res)
    res = re.sub("September", "09.", res)
    res = re.sub("Oktober", "10.", res)
    res = re.sub("November", "11.", res)
    res = re.sub("Dezember", "12.", res)
    res = res.split(" ")
    #print(len(res))
    r= res[0]
    if len(res)==3:
        r = ""
        for i in res:
            r = r + str(i)
    if "-" in r:
        l = r.split("-")
        day = l[2]
        month = l[1]
        year = l[0]
        r= str(day)+"."+str(month)+"."+str(year)
    return r

def wordcount_in_docs(dictionary, word):
    """ prints how often a word is found and in how many documents it is found"""
    found = False
    for i in dictionary:
        if dictionary[i] == word:
            found = True
            print("found word", str(dictionary[i]), str(dictionary.cfs[i]), "times")
            print("word",str(dictionary[i]),"appears in", str(dictionary.dfs[i]), "documents")
    if not found:
        print(word, str(0))

def wordlist_in_docs(dictionary, wordlist):
    """ prints how often a word is found and in how many documents it is found"""
    found = False
    for i in dictionary:
        if dictionary[i] in wordlist:
            found = True
            print("found word", str(dictionary[i]), str(dictionary.cfs[i]), "times")
            print("word",str(dictionary[i]),"appears in", str(dictionary.dfs[i]), "documents")
    if not found:
        print("not found")

def docnr_for_topic(model, corpus, topic_nr, min_prob):
    """ gets documents that contain a topic with the minimum probability
        model, corpus: LDA model and corpus
        topic_nr: the ID of the topic
        minprob: minimum probability for the topic in a document
        returns list of (doc-ID, prob)
     """
    topics = model.print_topics(num_topics=200)
    print(topics)
    docs_per_topic = [[] for _ in topics]
    for id, doc in enumerate(corpus):
        doc_topics = model.get_document_topics(doc, minimum_probability=min_prob)
        for topic_id, val in doc_topics:
            docs_per_topic[topic_id].append((id, val))
    docs = docs_per_topic[topic_nr]
    print(len(docs), "documents contain topic", str(topic_nr), "with at least", str(min_prob), "%.")
    return docs


def topic_in_docs(model, corpus, data_words, topic_nr, minprob, filename):
    """ creates csv-file with the data_words-ID and percentage
        model: the LDA model
        corpus: the model's corpus
        data_words: the lemmatized tokens for each document
        topic_nr: the ID of the topic
        minprob: minimum probability for the topic in a document
        filename: name of the csv-file that is created
    """
    docs = docnr_for_topic(model, corpus, topic_nr, minprob)
    for doc in docs:
        print("Tokens: " + str(data_words[doc[0]]))
        print("doc " + str(doc[0])+ ":" + str(doc[1]))

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["DocNr", "%"])
        for doc in docs:
            writer.writerow([doc[0], doc[1]])

def print_data_to_docs(readfile):
    """ prints metadata information to document-IDs in readfile[0] """
    with open(readfile, 'r') as file:
        file.readline()
        reader = csv.reader(file)
        rows = list(reader)
        with open('/Users/juliakarst/PycharmProjects/NewsLDA/data/all_articles.csv', 'r') as reference:
            reference.readline()
            reader2 = csv.reader(reference)
            rows2 = list(reader2)
            for row in rows:
                i = row[0]
                date = sub(rows2[int(i)+1][2])
                #print("data_word entry", i)
                #print("Tokens: " + str(data_words[int(i)]))
                #print(rows2[int(i)+1][0])
                print(rows2[int(i)+1][5])
                #print(date)


def create_doclist_for_word(word):
    """ takes a word and creates a list of document-IDs in which that word occurs"""
    i = 0
    listofdocs = []
    for doc in data_words:
        if word in doc:
            #print(i, doc)
            listofdocs.append(i)
        i = i + 1
    return listofdocs

def create_doclist_for_wordlist(wordlist):
    """ takes a word and creates a list of document-IDs in which that word occurs"""
    i = 0
    listofdocs = []
    for word in wordlist:
        for doc in data_words:
            if word in doc:
                #print(i, doc)
                listofdocs.append(i)
            i = i + 1
    return set(listofdocs)

def info_to_doc(file, doclist, resultfile):
    """ writes the information taken from file about the document-IDs given in doclist in resultfile
        file: csv-Datei mit Metadaten zu allen Dokumenten
        doclist: Liste, die Dokument-IDs enthält
        resultfile: Pfad für Ausgabedatei
    """
    towrite=[["Dokument-ID","Datum", "Quelle", "Titel"]]

    with open(file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        csvfile.readline()
        j = 1
        for row in csvreader:
            if (j - 2) in doclist:
                title = row[0]
                date = sub(row[2])
                source = row[5]
                print(str(j),",",date,",", source,",", title)
                towrite.append([str(j),date,source,title])
            j = j + 1
    writefile = open(resultfile, 'w')
    with writefile:
        writer = csv.writer(writefile)
        for row in towrite:
            writer.writerow(row)

def show_doc_attribution(model, corpus, data_words, i):
    """ prints topics, topic attribution and tokens
        model: the LDA model
        corpus: the model's corpus
        data_words: the lemmatized tokens for each document
        i: document ID
    """
    #pprint(model.print_topics()) #print topics with id
    #topics = model.print_topics()
    get_doc_topics = model.get_document_topics(corpus[i])
    print("Tokens: " + str(data_words[i]))
    print("Topic Attribution for Document "+str(i+1)+" from the corpus: " + str(get_doc_topics))


def topdocs_for_topic(model, corpus, t, top_n):
    """ returns the top_n documents that are associated with topic t
        model: the LDA model
        corpus: the model's corpus
        t: Topic ID
        top_n: number of documents
    """
    topics = model.print_topics(num_topics=150)
    docs_per_topic = [[] for _ in topics]

    for id, doc in enumerate(corpus):
        doc_topics = model.get_document_topics(doc)
        for topic_nr, value in doc_topics:
            docs_per_topic[topic_nr].append((id, value))

    for doc_list in docs_per_topic:
        doc_list.sort(key=lambda i: i[1], reverse=True)

    return(docs_per_topic[t][:top_n])

def convert_topics(model, top):
    """ takes only the words from the topics, discards the percentages"""
    topics = model.print_topics(num_topics=150,num_words=top)
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


#topics = model.print_topics(num_topics=20, num_words=15)
#for t in topics:
#    print(t)

topic_in_docs(model,corpus,data_words, 15, 0.1,'topic_olympia.csv')
#print_data_to_docs('topic_olympia.csv')

#show the attribution of topics to one document
#print(data_words[0])
#show_doc_attribution(model, corpus, data_words, 0)

#print the top documents that belong to one topic
#for i in topdocs_for_topic(model, corpus, 15, 10):
#    print("Dokument Nr. ", str(i[0]+2))
#    print(data_words[i[0]])
#    print(i)

#wordcount_in_docs(news_dictionary, "corona")
#listofdocs = create_doclist_for_word("corona")
#info_to_doc('/Users/juliakarst/PycharmProjects/NewsLDA/data/all_articles.csv', listofdocs, "corona.csv")

