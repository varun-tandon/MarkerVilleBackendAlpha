
# coding: utf-8

# Misc imports

# In[4]:



import os

import matplotlib.pyplot as plt
from cPickle import load
import cPickle
import numpy as np


from snorkel.snorkel import SnorkelSession
from snorkel.parser import XMLMultiDocPreprocessor, CorpusParser
from snorkel.parser.spacy_parser import Spacy
from snorkel.parser.corenlp import StanfordCoreNLPServer
from snorkel.models import Document, Sentence, Candidate, candidate_subclass
from snorkel.candidates import Ngrams, CandidateExtractor
from snorkel.viewer import SentenceNgramViewer
from snorkel.annotations import LabelAnnotator, load_gold_labels, FeatureAnnotator, save_marginals, load_marginals
from snorkel.learning import SparseLogisticRegression, GenerativeModel, RandomSearch
from snorkel.learning.structure import DependencySelector
from snorkel.learning.utils import MentionScorer
# from snorkel.contrib.rnn import reRNN

import matchers
import LF
from candidate_adjective_fixer import *
from load_external_annotations_new import load_external_labels

session = SnorkelSession()

BiomarkerCondition = candidate_subclass('BiomarkerCondition', ['biomarker', 'condition'])



# Helper functions

# In[ ]:


#------------------
# Helper Functions
#------------------

def grabCandidates(extractor, schema):
    # Candidate Counts
    for k, sents in enumerate([train_sents, dev_sents, test_sents]):
        extractor.apply(sents, split=k, clear=False)
        print "Number of candidates: ", session.query(schema).filter(schema.split == k).count()
        session.commit()
        
    train_cands = session.query(schema).filter(
        schema.split == 0).all()
    dev_cands = session.query(schema).filter(
        schema.split == 1).all()
    test_cands = session.query(schema).filter(
        schema.split == 2).all()

    return [train_cands, dev_cands, test_cands]


# In[ ]:


#-----------------------
# Setup & Preprocessing
#-----------------------

# Instantiate the Session
session = SnorkelSession()

# Doc Preprocessing
file_path = 'articles/training.xml'
train_preprocessor = XMLMultiDocPreprocessor(
    path=file_path,
    doc='.//article',
    text='.//front/article-meta/abstract/p/text()',
    id='.//front/article-meta/article-id/text()'
)

file_path = 'articles/development.xml'
dev_preprocessor = XMLMultiDocPreprocessor(
    path=file_path,
    doc='.//document',
    text='.//passage/text/text()',
    id='.//id/text()'
)

file_path = 'articles/testcorpus.xml'
test_preprocessor = XMLMultiDocPreprocessor(
    path=file_path,
    doc='.//document',
    text='.//passage/text/text()',
    id='.//id/text()'
)

# Parsing
# corenlp_server = StanfordCoreNLPServer(version="3.6.0", num_threads=4, port=12348)
# corpus_parser = CorpusParser(corenlp_server, parser=Spacy())
corpus_parser = CorpusParser(parser=Spacy())
# corpus_parser = CorpusParser()

# Note: Parallelism can be run with a Postgres DBMS, but not SQLite
corpus_parser.apply(list(train_preprocessor))
corpus_parser.apply(list(dev_preprocessor), clear=False)
corpus_parser.apply(list(test_preprocessor), clear=False)



# In[ ]:


# Retrieving Stable IDs for each of the candidate sentences
with open('articles/doc_ids.pkl', 'rb') as f:
    train_ids, dev_ids, test_ids = load(f)

train_ids, dev_ids, test_ids = set(train_ids), set(dev_ids), set(test_ids)
train_sents, dev_sents, test_sents = set(), set(), set()
docs = session.query(Document).order_by(Document.name).all()


# Assigning each sentence to {train,dev,test}-set based on Stable ID
for i, doc in enumerate(docs):
    for s in doc.sentences:
        if doc.name in train_ids:
            train_sents.add(s)
        elif doc.name in dev_ids:
            dev_sents.add(s)
        elif doc.name in test_ids:
            test_sents.add(s)
        else:
            raise Exception(
                'ID <{0}> not found in any id set'.format(doc.name))


# In[ ]:


#----------------------
# Candidate Extraction
#----------------------

# Defining the Candidate Schemas
# BiomarkerCondition = candidate_subclass('BiomarkerCondition', ['biomarker', 'condition'])

# BiomarkerDrug = candidate_subclass('BiomarkerDrug', ['biomarker', 'drug'])
# BiomarkerMedium = candidate_subclass('BiomarkerMedium', ['biomarker', 'medium'])
# BiomarkerType = candidate_subclass('BiomarkerType', ['biomarker', 'typ3'])
# # BiomarkerLevelUnit = candidate_subclass('BiomarkerLevelUnit', ['biomarker', 'level', 'unit'])
#can eventually add MEASUREMENT and COHORT SIZE among other entities

# N-grams: the probabilistic search space of our entities
biomarker_ngrams = Ngrams(n_max=1)
condition_ngrams = Ngrams(n_max=7)
# drug_ngrams = Ngrams(n_max=5)
# medium_ngrams = Ngrams(n_max=5)
# type_ngrams = Ngrams(n_max=5)  # <--- Q: should we cut these down?
# # level_ngrams = Ngrams(n_max=1)
# unit_ngrams = Ngrams(n_max=1)

# Construct our Matchers
bMatcher = matchers.getBiomarkerMatcher()
cMatcher = matchers.getConditionMatcher()
# dMatcher = matchers.getDrugMatcher()
# mMatcher = matchers.getMediumMatcher()
# tMatcher = matchers.getTypeMatcher()
# lMatcher = matchers.getLevelMatcher()
# uMatcher = matchers.getUnitMatcher()

# Building the CandidateExtractors
candidate_extractor_BC = CandidateExtractor(BiomarkerCondition, [biomarker_ngrams, condition_ngrams], [bMatcher, cMatcher])
# candidate_extractor_BD = CandidateExtractor(BiomarkerDrug, [biomarker_ngrams, drug_ngrams], [bMatcher, dMatcher])
# candidate_extractor_BM = CandidateExtractor(BiomarkerMedium, [biomarker_ngrams, medium_ngrams], [bMatcher, mMatcher])
# candidate_extractor_BT = CandidateExtractor(BiomarkerType, [biomarker_ngrams, type_ngrams], [bMatcher, tMatcher])
# candidate_extractor_BLU = CandidateExtractor(BiomarkerLevelUnit, [biomarker_ngrams, level_ngrams, unit_ngrams], [bMatcher, lMatcher, uMatcher])

# List of Candidate Sets for each relation type: [train, dev, test]
cands_BC = grabCandidates(candidate_extractor_BC, BiomarkerCondition)
# cands_BD = grabCandidates(candidate_extractor_BD, BiomarkerDrug)
# cands_BM = grabCandidates(candidate_extractor_BM, BiomarkerMedium)
# cands_BT = grabCandidates(candidate_extractor_BT, BiomarkerType)
# cands_BLU = grabCandidates(candidate_extractor_BLU, BiomarkerLevelUnit)



# In[ ]:


session.rollback()
print "Number of dev BC candidates without adj. boosting: ", len(cands_BC[1])
add_adj_candidate_BC(session, BiomarkerCondition, cands_BC[1], 0)
# fix_specificity(session, BiomarkerCondition, cands_BC[1])
print "Number of dev BC candidates with adj. boosting: ", session.query(BiomarkerCondition).filter(BiomarkerCondition.split == 1).count()
session.commit()


# In[ ]:


from LF import *
LFs_BC = [LF_markerDatabase, LF_keyword, LF_distance, LF_abstract_titleWord, LF_single_letter,
          LF_auxpass, LF_known_abs, LF_same_thing_BC, LF_common_1000, LF_common_2000]


# In[ ]:


from snorkel.annotations import LabelAnnotator
BC_labeler = LabelAnnotator(lfs=LFs_BC)


# In[ ]:


np.random.seed(1701)
get_ipython().magic(u'time L_train_BC = BC_labeler.apply(split=0)')
L_train_BC


# In[ ]:


get_ipython().magic(u'time L_train_BC = BC_labeler.load_matrix(session, split=0)')
L_train_BC


# In[ ]:


L_train_BC.get_candidate(session, 0)


# In[ ]:


L_train_BC.get_key(session, 0)


# In[ ]:


from snorkel.learning import GenerativeModel

gen_model = GenerativeModel()
gen_model.train(L_train_BC, epochs=100, decay=0.95, step_size=0.1 / L_train_BC.shape[0], reg_param=1e-6)


# In[ ]:


gen_model.weights.lf_accuracy


# In[ ]:


train_marginals = gen_model.marginals(L_train_BC)


# In[ ]:


import matplotlib.pyplot as plt
plt.hist(train_marginals, bins=20)
plt.show()


# In[ ]:


L_dev = BC_labeler.apply_existing(split=1)


# In[ ]:


from snorkel.annotations import save_marginals
get_ipython().magic(u'time save_marginals(session, L_train_BC, train_marginals)')


# In[5]:


from snorkel.annotations import load_marginals

train_marginals = load_marginals(session, split=0)


# In[7]:


train_cands = session.query(BiomarkerCondition).filter(BiomarkerCondition.split == 0).order_by(BiomarkerCondition.id).all()
dev_cands   = session.query(BiomarkerCondition).filter(BiomarkerCondition.split == 1).order_by(BiomarkerCondition.id).all()
test_cands  = session.query(BiomarkerCondition).filter(BiomarkerCondition.split == 2).order_by(BiomarkerCondition.id).all()


# In[8]:


from snorkel.annotations import load_gold_labels
load_external_labels(session, BiomarkerCondition, 'Biomarker', 'Condition', 'articles/disease_gold_labels.tsv', dev_cands, annotator_name='gold')
load_external_labels(session, BiomarkerCondition, 'Biomarker', 'Condition', 'articles/disease_test_labels.tsv', test_cands, annotator_name='gold')

L_gold_dev  = load_gold_labels(session, annotator_name='gold', split=1)
L_gold_test = load_gold_labels(session, annotator_name='gold', split=2)




# In[ ]:


print len(train_cands)
print len(dev_cands)


# In[9]:


from snorkel.learning.disc_models.rnn import reRNN

train_kwargs = {
    'lr':         0.01,
    'dim':        50,
    'n_epochs':   10,
    'dropout':    0.25,
    'print_freq': 1,
    'max_sentence_length': 100
}

lstm = reRNN(seed=1701, n_threads=None)
lstm.train(train_cands, train_marginals, X_dev=dev_cands, Y_dev=L_gold_dev, **train_kwargs)


# In[10]:


p, r, f1 = lstm.score(test_cands, L_gold_test)
print("Prec: {0:.3f}, Recall: {1:.3f}, F1 Score: {2:.3f}".format(p, r, f1))


# In[11]:


tp, fp, tn, fn = lstm.error_analysis(session, test_cands, L_gold_test)


# In[ ]:


lstm.save_marginals(session, test_cands)


# In[23]:


predictions = lstm.predictions(train_cands)


# In[26]:


i = 0
for prediction in predictions: 
    if(prediction == 1):
        i+=1
print i


# In[24]:


i = 0
while( i< len(train_cands)):
    print("Candidate: {}. Prediction: {}").format(train_cands[i], predictions[i])
    i += 1


# In[27]:


lstm.save()


# In[ ]:




