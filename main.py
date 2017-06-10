import numpy as np
#import re
#import ngram
import nskipgram
import featurize
import lemmatize
#import posLimiter
#import create_lookup
#import stopwords
#from stemming.porter2 import stem
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

# uni+bi = .62839673913
#uni + bi + poslimit (adj) = 0.624320652174
#uni + nskipgram =  0.641983695652
#uni + bi + poslimit (adj) + nskipgram =  0.627038043478

#uni + bi + freq = 0.544157608696
#uni + nskipgram + freq = 0.531929347826
#uni + posLimiter + freq = 0.447010869565
#uni + nskipgram + posLimiter + freq = 0.45652173913

##starts stanford server
#cd stanford-corenlp-full-2016-10-31
#java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 10000

eng_stopwords = []
with open('C:\\Users\\rushilcd\\Desktop\\On-Site Docs\\stanford-corenlp-full-2016-10-31\\patterns\\stopwords.txt') as inputfile:
    for line in inputfile:
        eng_stopwords.append(line.strip()) 
        
# Load and take only relevant fields among input data
sents = np.loadtxt('C:\\Users\\rushilcd\\Desktop\\On-Site Docs\\Data\\original_rt_snippets', dtype = str, delimiter = '\n')
#sent = [x.split() for x in sents]
label = np.loadtxt('C:\\Users\\rushilcd\\Desktop\\On-Site Docs\\Data\\labels', dtype = str, delimiter = '\n')

##stemming
execfile('lemmatize.py')
sent = [lemmatize(y) for y in sents]
sent = [x for x in sent if x!=None]

###adjective limiter
#execfile('posLimiter.py')
#sent = [posLimiter(x) for x in sent]

## These are the relevant fields
concat = zip(sent,label)
concat = [x for x in concat if x[0]!=None]
train = concat[:9000]
test = concat[9000:]

sent = zip(*train)[0]
label = zip(*train)[1]
# Process the data to remove stopwords, capital letters and other symbols
sent = [[y.lower() for y in x if y.lower() not in eng_stopwords] for x in sent]  


concat = zip(label,sent)

# This function prepares ngrams of any order n
execfile('ngram.py')
execfile('nskipgram.py')
# This function updates the dictionary 'dict' with the counts of E,C,N 

#bigram = [ngram(2,x) for x in sent]
skipgram = [nskipgram(len(x)-2,2,x) for x in sent]
concat = zip(label,skipgram)

from sets import Set
lookup = Set([])

def create_lookup(itm,s):
	s.add(itm)
	return None


nothing = [[create_lookup(y,lookup) for y in x] for x in sent]
#nothing = [[create_lookup(y,lookup) for y in x] for x in bigram]
nothing = [[create_lookup(y,lookup) for y in x] for x in skipgram]

#lookup = {}
#execfile('create_lookup.py')
#
#
##nothing = [create_lookup(sent[i],bigram[i],label[i],lookup) for i in range(len(sent))]
#nothing = [create_lookup(sent[i],skipgram[i],label[i],lookup) for i in range(len(sent))]


lookup = zip(lookup, range(len(lookup)))
lookup = {x[0]:x[1] for x in lookup}
#lookup = {x[0][0]:(x[0][1],x[1]) for x in lookup}

########################################## Feature Vector Generation ######################################

# presence not counts
#concat = zip(range(len(sent)),sent, bigram)
concat = zip(range(len(sent)),sent, skipgram)


row = []
col = []
data = []

# This function generates feature vector in sparse format for every sentence pair 
execfile('featurize.py')

nothing = [featurize(x[0],x[1],x[2]) for x in concat]
data = [1]*len(row)

###Counts not presence
#def featurize(index,unigram,bigram):
#    for itm in unigram:
#    		try:
#    			col.extend([5*lookup[itm][1] + i  for i in range(5)])
#    			row.extend([index]*5)
#    			data.extend(lookup[itm][0])
#    		except Exception:
#    			continue
#            
#    for itm in bigram:
#    		try:
#    			col.extend([5*lookup[itm][1] + i for i in range(5)])
#    			row.extend([index]*5)
#    			data.extend(lookup[itm][0])
#    		except Exception:
#    			continue
#    return None

#nothing = [featurize(i,sent[i],bigram[i]) for i in range(len(sent))]
#nothing = [featurize(i,sent[i],skipgram[i]) for i in range(len(sent))]

from scipy.sparse import bsr_matrix
X = bsr_matrix((data, (row, col)))
y = [4 if x=='Very positive' else 3 if x=='Positive' else 2 if x=='Neutral' else 1 if 'Negative' else 0 for x in label]


########################################## SVM classifier  ##############################################
from sklearn import svm
clf = svm.LinearSVC() #decision_function_shape='ovo'
accuracies = []
#import time
#start_time = time.time()
clf.fit(X,y)
out = clf.predict(X)
#print 'elapsed_time = '+str((time.time() - start_time))+' secs'
accuracy = (sum([1 for x in y-out if x==0])+0.0)/len(y)
print ('accuracy = ' + str(accuracy))
accuracies.append(accuracy)
######################################### Naive Bayes CLassifier #########################################
#from sklearn.naive_bayes import MultinomialNB
#clf = MultinomialNB()
#
#import time
#start_time = time.time()
#clf.fit(X,y)
#out = clf.predict(X)
#print ('elapsed_time = '+str((time.time() - start_time))+' secs')
#
#accuracy = (sum([1 for x in y-out if x==0])+0.0)/len(y)
#
#print ('accuracy = '+ str(accuracy))
#accuracies.append(accuracy)
######################################### Accuracy on test data ############################################
del sent
del label
#del bigram
del skipgram
del concat
# These are the relevant fields
sent = zip(*test)[0]
label = zip(*test)[1]


# Process the data to remove stopwords, capital letters and other symbols
sent = [[y.lower() for y in x if y.lower() not in eng_stopwords] for x in sent]

#bigram = [ngram(2,x) for x in sent]
skipgram = [nskipgram(len(x)-2,2,x) for x in sent]

#concat = zip(range(len(sent)),sent,bigram)
concat = zip(range(len(sent)),sent,skipgram)

row = []
col = []
data = []

nothing = [featurize(x[0],x[1],x[2]) for x in concat]
#nothing = [featurize(i,sent[i],bigram[i]) for i in range(len(sent))]
#nothing = [featurize(i,sent[i],skipgram[i]) for i in range(len(sent))]

data = [1]*len(row)

if max(col)<X.shape[1]-1:
	row.append(max(row))
	col.append(X.shape[1]-1)
	data.append(0)

X_test = bsr_matrix((data, (row, col)))
out_test = clf.predict(X_test)

y_test = [4 if x=='Very positive' else 3 if x=='Positive' else 2 if x=='Neutral' else 1 if 'Negative' else 0 for x in label]

accuracy = (sum([1 for x in y_test-out_test if x==0])+0.0)/len(y_test)

print ('accuracy = '+ str(accuracy))
accuracies.append(accuracy)
#print('Accuracies in order of SVM Train, NB Train, Test' + ', '.join([str(acc) for acc in accuracies]))