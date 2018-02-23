

import os
import matplotlib.pyplot as plt
from six.moves.cPickle import load
import cPickle
import numpy as np
from unidecode import unidecode
from snorkel import SnorkelSession
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
from snorkel.parser import XMLMultiDocPreprocessor
from snorkel import SnorkelSession
from snorkel.parser import CorpusParser

import matchers
import LF
from candidate_adjective_fixer import *
from load_external_annotations_new import load_external_labels
from snorkel.learning.disc_models.rnn import reRNN
from candidate_adjective_fixer import *


def parse_corpus(to_process_file):
    file_path = to_process_file
    doc_preprocessor = XMLMultiDocPreprocessor(
        path= file_path,
        doc='.//Article',
        text='./text/text()',
        id='./article-id/text()'
    )
    corpus_parser = CorpusParser(parser=Spacy())
    corpus_parser.apply(list(doc_preprocessor))
    return corpus_parser

def run(candidate1, candidate2, pairing_name, cand1_ngrams, cand2_ngrams, cand1Matcher, cand2Matcher, model_name, output_file_name, corpus_parser):
    print "Started"
    session = SnorkelSession()

    # The following line is for testing only. Feel free to ignore it.

   

    candidate_pair = candidate_subclass(pairing_name, [candidate1, candidate2])

    sentences = set()
    docs = session.query(Document).order_by(Document.name).all()
    for doc in docs:
        for s in doc.sentences: 
            sentences.add(s)


    cand_1_ngrams = Ngrams(n_max=cand1_ngrams)
    # condition_ngrams = Ngrams(n_max=7)
    cand_2_ngrams = Ngrams(n_max=cand2_ngrams)
    # medium_ngrams = Ngrams(n_max=5)
    # type_ngrams = Ngrams(n_max=5)  # <--- Q: should we cut these down?
    # # level_ngrams = Ngrams(n_max=1)
    # unit_ngrams = Ngrams(n_max=1)

    # Construct our Matchers


    # cMatcher = matchers.getConditionMatcher()
    # mMatcher = matchers.getMediumMatcher()
    # tMatcher = matchers.getTypeMatcher()
    # lMatcher = matchers.getLevelMatcher()
    # uMatcher = matchers.getUnitMatcher()

    # Building the CandidateExtractors
    # candidate_extractor_BC = CandidateExtractor(BiomarkerCondition, [biomarker_ngrams, condition_ngrams], [bMatcher, cMatcher])
    candidate_extractor = CandidateExtractor(candidate_pair, [cand_1_ngrams, cand_2_ngrams], [cand1Matcher, cand2Matcher])
    # candidate_extractor_BM = CandidateExtractor(BiomarkerMedium, [biomarker_ngrams, medium_ngrams], [bMatcher, mMatcher])
    # candidate_extractor_BT = CandidateExtractor(BiomarkerType, [biomarker_ngrams, type_ngrams], [bMatcher, tMatcher])
    # candidate_extractor_BLU = CandidateExtractor(BiomarkerLevelUnit, [biomarker_ngrams, level_ngrams, unit_ngrams], [bMatcher, lMatcher, uMatcher])

    # List of Candidate Sets for each relation type: [train, dev, test]
    candidate_extractor.apply(sentences, split=4, clear=True)
    cands = session.query(candidate_pair).filter(candidate_pair.split == 4).order_by(candidate_pair.id).all()
    session.commit()
    # cands_BD = grabCandidates(candidate_extractor_BD, BiomarkerDrug)
    # cands_BM = grabCandidates(candidate_extractor_BM, BiomarkerMedium)
    # cands_BT = grabCandidates(candidate_extractor_BT, BiomarkerType)
    # cands_BLU = grabCandidates(candidate_extractor_BLU, BiomarkerLevelUnit)


    if(len(cands)) == 0:
        print "No Candidates Found"
        return 
    if(pairing_name == 'BiomarkerCondition'):    
        # session.rollback()
        # print "Number of dev BC candidates without adj. boosting: ", len(cands_BC[1])
        add_adj_candidate_BC(session, candidate_pair, cands, 4)
        # fix_specificity(session, BiomarkerCondition, cands_BC[1])
        # print "Number of dev BC candidates with adj. boosting: ", session.query(BiomarkerCondition).filter(BiomarkerCondition.split == 4).count()
        session.commit()


    lstm = reRNN(seed=1701, n_threads=None)

    lstm.load(model_name)

    predictions = lstm.predictions(cands)
    output_file = open(output_file_name, 'wb')
    import csv 
    csvWriter = csv.writer(output_file)
    csvWriter.writerow(['doc_id', 'sentence', candidate1, candidate2, 'prediction'])
    for i in range(len(cands)):
        doc_string = 'PMC' + str(cands[i].get_parent().get_parent())[9:]
        sentence_string = cands[i].get_parent().text
        cand_1_string = cands[i].get_contexts()[0].get_span()
        cand_2_string = cands[i].get_contexts()[1].get_span()
        prediction = predictions[i]
        csvWriter.writerow([unidecode(doc_string), unidecode(sentence_string), unidecode(cand_1_string), unidecode(cand_2_string), prediction])




