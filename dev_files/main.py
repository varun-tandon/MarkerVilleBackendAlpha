# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import os
import matplotlib.pyplot as plt
# from six.moves.cPickle import load
import cPickle
import numpy as np

from snorkel import SnorkelSession
from snorkel.parser import XMLMultiDocPreprocessor, CorpusParser
from snorkel.models import Document, Sentence, Candidate, candidate_subclass
from snorkel.candidates import Ngrams, CandidateExtractor
from snorkel.viewer import SentenceNgramViewer
from snorkel.annotations import LabelAnnotator, load_gold_labels, FeatureAnnotator, save_marginals, load_marginals
from snorkel.learning import SparseLogisticRegression, GenerativeModel, RandomSearch, ListParameter, RangeParameter
from snorkel.learning.structure import DependencySelector
from snorkel.learning.utils import MentionScorer
from snorkel.contrib.rnn import reRNN

import matchers
import LF
from candidate_adjective_fixer import *
from load_external_annotations_new import load_external_labels


#------------------
# Helper Functions
#------------------

def grabCandidates(extractor, schema):
    # Candidate Counts
    for k, sents in enumerate([train_sents, dev_sents, test_sents]):
        extractor.apply(sents, split=k)
        print "Number of candidates: ", session.query(schema).filter(schema.split == k).count()

    train_cands = session.query(schema).filter(
        schema.split == 0).all()
    dev_cands = session.query(schema).filter(
        schema.split == 1).all()
    test_cands = session.query(schema).filter(
        schema.split == 2).all()

    return [train_cands, dev_cands, test_cands]


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
corpus_parser = CorpusParser()

# Note: Parallelism can be run with a Postgres DBMS, but not SQLite
corpus_parser.apply(list(train_preprocessor))
corpus_parser.apply(list(dev_preprocessor), clear=False)
corpus_parser.apply(list(test_preprocessor), clear=False)


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


#----------------------
# Candidate Extraction
#----------------------

# Defining the Candidate Schemas
BiomarkerCondition = candidate_subclass(
    'BiomarkerCondition', ['biomarker', 'condition'])
BiomarkerDrug = candidate_subclass('BiomarkerDrug', ['biomarker', 'drug'])
BiomarkerMedium = candidate_subclass(
    'BiomarkerMedium', ['biomarker', 'medium'])


# N-grams: the probabilistic search space of our entities
biomarker_ngrams = Ngrams(n_max=1)
condition_ngrams = Ngrams(n_max=7)
drug_ngrams = Ngrams(n_max=5)
medium_ngrams = Ngrams(n_max=5)
type_ngrams = Ngrams(n_max=5)  # <--- Q: should we cut these down?


# Construct our Matchers
bMatcher = matchers.getBiomarkerMatcher()
cMatcher = matchers.getDiseaseMatcher()
dMatcher = matchers.getDrugMatcher()
mMatcher = matchers.getMediumMatcher()
tMatcher = matchers.getTypeMatcher()


# Building the CandidateExtractors
candidate_extractor_BC = CandidateExtractor(
    BiomarkerCondition, [biomarker_ngrams, condition_ngrams], [bMatcher, cMatcher])
candidate_extractor_BD = CandidateExtractor(
    BiomarkerDrug, [biomarker_ngrams, drug_ngrams], [bMatcher, dMatcher])
candidate_extractor_BM = CandidateExtractor(
    BiomarkerMedium, [biomarker_ngrams, medium_ngrams], [bMatcher, mMatcher])
candidate_extractor_BT = CandidateExtractor(
    BiomarkerType, [biomarker_ngrams, type_ngrams], [bMatcher, tMatcher])


# List of Candidate Sets for each relation type: [train, dev, test]
cands_BC = grabCandidates(candidate_extractor_BC, BiomarkerCondition)
cands_BD = grabCandidates(candidate_extractor_BD, BiomarkerDrug)
cands_BM = grabCandidates(candidate_extractor_BM, BiomarkerMedium)
cands_BT = grabCandidates(candidate_extractor_BT, BiomarkerType)


# Adjective Boosting - BC
print "Number of dev BC candidates without adj. boosting: ", len(cands_BC[1])
add_adj_candidate_BC(session, BiomarkerCondition, cands_BC[1])
fix_specificity(session, BiomarkerCondition, cands_BC[1])
print "Number of dev BC candidates with adj. boosting: ", session.query(BiomarkerCondition).filter(BiomarkerCondition.split == 1).count()
session.commit()

# Adjective Boosting - BD
print "Number of dev BD candidates without adj. boosting: ", len(cands_BD[1])
add_adj_candidate_BD(session, BiomarkerDrug, cands_BD[1])
print "Number of dev BD candidates with adj. boosting: ", session.query(BiomarkerDrug).filter(BiomarkerDrug.split == 1).count()
session.commit()

# Adjective Boosting - BM
print "Number of dev BM candidates without adj. boosting: ", len(cands_BM[1])
add_adj_candidate_BM(session, BiomarkerMedium, dev_cands)
print "Number of dev BM candidates with adj. boosting: ", session.query(BiomarkerMedium).filter(BiomarkerMedium.split == 1).count()
session.commit()

# Adjective Boosting - BT (none as of now)


#-------------------------------------------
# External Gold Labels & Labeling Functions
#-------------------------------------------

# Labling Functions
LFs_BC = [LF_markerDatabase, LF_keyword, LF_distance, LF_abstract_titleWord, LF_single_letter,
          LF_auxpass, LF_known_abs, LF_same_thing_BC, LF_common_1000, LF_common_2000]
LFs_BD = [LF_colon, LF_known_abs, LF_single_letter,
          LF_roman_numeral, LF_common_2000, LF_same_thing_BD]
LFs_BM = [LF_distance_far, LF_colon, LF_known_abs, LF_single_letter, LF_roman_numeral, LF_common_2000, LF_same_thing]
LFs_BT = [LF_colon, LF_known_abs, LF_single_letter, LF_roman_numeral, LF_common_2000, LF_same_thing]

labeler_BC = LabelAnnotator(lfs=LFs_BC)
labeler_BD = LabelAnnotator(lfs=LFs_BD)
labeler_BM = LabelAnnotator(lfs=LFs_BM)
labeler_BT = LabelAnnotator(lfs=LFs_BT)


# Training
L_train_BC = labeler_BC.apply(split=0)
L_train_BD = labeler_BD.apply(split=0)
L_train_BM = labeler_BM.apply(split=0)
L_train_BT = labeler_BT.apply(split=0)
L_train_BC
L_train_BD
L_train_BM
L_train_BT


# Labeling Function Performance - Coverage, Overlaps, Conflicts
L_train_BC.lf_stats(session)
L_train_BD.lf_stats(session)
L_train_BM.lf_stats(session)
L_train_BT.lf_stats(session)


# Analyzing Dependencies
Ldeps = []
for L in [L_train_BC, L_train_BD, L_train_BD, L_train_BD]:
    ds = DependencySelector()
    deps = ds.select(L, threshold=0.1)
    len(deps)
    Ldeps.append(deps)


gen_model = GenerativeModel(lf_propensity=True)
gen_model.train(
    L_train, deps=deps, decay=0.95, step_size=0.1 / L_train.shape[0], reg_param=0.0
)
train_marginals = gen_model.marginals(L_train)
plt.hist(train_marginals, bins=20)
plt.show()
gen_model.learned_lf_stats()
save_marginals(session, L_train, train_marginals)
load_external_labels(session, BiomarkerCondition, 'Biomarker', 'Condition',
                     'articles/disease_gold_labels.tsv', dev_cands, annotator_name='gold')


L_gold_dev = load_gold_labels(session, annotator_name='gold', split=1)
L_gold_dev
L_dev = labeler.apply_existing(split=1)
_ = gen_model.score(session, L_dev, L_gold_dev)
L_dev.lf_stats(session, L_gold_dev, gen_model.learned_lf_stats()['Accuracy'])


labeled = []
for c in session.query(BiomarkerCondition).filter(BiomarkerCondition.split == 1).all():
    if LF_markerDatabase(c) == 1:
        labeled.append(c)
SentenceNgramViewer(labeled, session, n_per_page=3)


# Load dev labels and convert to [0, 1] range
L_gold_dev = load_gold_labels(session, annotator_name='gold', split=1)
dev_labels = (np.ravel(L_gold_dev.todense()) + 1) / 2


# Feature Extraction
featurizer = FeatureAnnotator()
F_train = featurizer.apply(split=0)
F_train
F_dev = featurizer.apply_existing(split=1)
F_test = featurizer.apply_existing(split=2)


train_marginals = load_marginals(session, F_train, split=0)
disc_model = SparseLogisticRegression()


# Searching over learning rate
rate_param = RangeParameter('lr', 1e-6, 1e-2, step=1, log_base=10)
l1_param = RangeParameter('l1_penalty', 1e-6, 1e-2, step=1, log_base=10)
l2_param = RangeParameter('l2_penalty', 1e-6, 1e-2, step=1, log_base=10)
searcher = RandomSearch(session, disc_model, F_train, train_marginals, [
    rate_param, l1_param, l2_param], n=20)


L_gold_dev = load_gold_labels(session, annotator_name='gold', split=1)
L_gold_dev
np.random.seed(1701)
searcher.fit(F_dev, L_gold_dev, n_epochs=50,
             rebalance=0.5, print_freq=25)


load_external_labels(session, BiomarkerCondition, 'Biomarker', 'Condition',
                     'articles/disease_test_labels.tsv', test_cands, annotator_name='gold')
L_gold_test = load_gold_labels(session, annotator_name='gold', split=2)
L_gold_test
tp, fp, tn, fn = disc_model.score(session, F_test, L_gold_test)


# Training the RNN - our Discriminative Model
train_kwargs = {
    'lr':         0.01,
    'dim':        100,
    'n_epochs':   50,
    'dropout':    0.5,
    'rebalance':  0.25,
    'print_freq': 5
}

lstm = reRNN(seed=1701, n_threads=None)
lstm.train(train_cands, train_marginals, dev_candidates=dev_cands,
           dev_labels=dev_labels, **train_kwargs)
lstm.save("biomarkercondition.lstm")
