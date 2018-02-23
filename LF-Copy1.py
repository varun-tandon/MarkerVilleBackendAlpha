import re
from snorkel.lf_helpers import *
from snorkel.lf_helpers import get_sent_candidate_spans
from snorkel.lf_helpers import get_left_tokens, get_right_tokens
from random import randint
import cPickle
import unicodedata
from PyDictionary import PyDictionary

#-----------------------------
# Distant Supervision Loading
#-----------------------------

keyWords = ["associate", "express", "marker", "biomarker", "elevated", "decreased",
            "correlation", "correlates", "found", "diagnose", "variant", "appear",
            "connect", "relate", "exhibit", "indicate", "signify", "show", "demonstrate",
            "reveal", "suggest", "evidence", "elevation", "indication", "diagnosis",
            "variation", "modification", "suggestion", "link", "derivation", "denote",
            "denotation", "demonstration", "magnification", "depression", "boost", "level",
            "advance", "augmentation", "lessening", "enhancement", "expression", "buildup",
            "diminishing", "diminishment", "reduction", "drop", "dwindling", "lowering"]

negationWords = ["not", "nor", "neither"]

sentence_keyword_lemmas = ["contain", "collect",
                           "find", "sample", "fluid", "tissue", "detection"]

toAdd = []
for keyword in keyWords:
    syns = (PyDictionary().synonym(keyword))
    if not syns == None:
        for syn in syns:
            if not syn in keyWords and not syn in toAdd:
                toAdd.append(syn)
for word in toAdd:
    keyWords.append(word)

markerDatabase = []
with open('databases/markerData.pickle', 'rb') as f:
    markerDatabase = cPickle.load(f)

knowAbbreviations = []
with open('databases/abbreviations.com.pkl', 'rb') as f:
    knowAbbreviations = cPickle.load(f)

with open('databases/common1000.pkl', 'rb') as f:
    common1000 = cPickle.load(f)

with open('databases/common2000.pkl', 'rb') as f:
    common2000 = cPickle.load(f)


"""
Labeling Functions:
--------------------
Get zip file with given data_id. Downloads to memory and returns a Python ZipFile by default.
When dealing with larger files where it may not be desired to load the entire file into memory,
specifying `file_path` will enable the file to be downloaded locally.


Parameters
----------
c: Candidate (Snorkel Object)
    Candidate Relation from the CandidateSet is applied to the definied function as a heuristic.

Returns
----------
vote: {-1,0,1}
    A classification vote on the nature of the relation based on rules we specify in teh function.
    -1 = FALSE
    0 = Abstain
    1 = TRUE

"""


#-------------------------
# Biomarker-Condition LFs
#-------------------------

def LF_distance(c):
    try:
        x = 0
        for thing in (get_between_tokens(c)):
            x += 1
        if(x > 8):
            return -1
        else:
            return 1
    except:
        print "Threw out: {}".format(c)

def LF_markerDatabase(c):
    try:
        if(c.biomarker.get_span() in markerDatabase):
            return 1
        else:
            return 0
    except:
        print "Threw out: {}".format(c)


def LF_abstract_titleWord(c):
    try:
        words_in_between = []
        for thing in get_between_tokens(c):
            words_in_between.append(thing)
        if(len(words_in_between) > 1 and words_in_between[0] == ":"):
            return -1
    except:
        print "Threw out: {}".format(c)


def LF_single_letter(c):
    try:
        if(len(c.biomarker.get_span()) < 2):
            return -1
        else:
            return 0
    except:
        print "Threw out: {}".format(c)

def LF_keyword(c):
    try:
        for keyword in keyWords:
            if(keyword in get_between_tokens(c)):
                if("not" in get_between_tokens(c)):
                    return -1
                else:
                    return 1
        return 0
    except:
        print "Threw out: {}".format(c)

def LF_auxpass(c):
    try:
        if not 'auxpass' in get_between_tokens(c, attrib='dep_labels'):
            return -1
        else:
            return 1
    except:
        print "Threw out: {}".format(c)

def LF_known_abs(c):
    try:
        if(c.biomarker.get_span() in knowAbbreviations):
            return -1
    except:
        print "Threw out: {}".format(c)
def LF_same_thing_BC(c):
    try:


        if( not type(c) == 'snorkel.models.candidate.BiomarkerCondition'):
            return
        if(c.biomarker.get_span() == c.condition.get_span()):
            return -1
    except:
        print "Threw out: {}".format(c)
def LF_common_1000(c):
    try:
        if(not type(c) == 'snorkel.models.candidate.BiomarkerCondition'):
            return
        if(c.condition.get_span() in common1000):
            return -1
    except:
        print "Threw out: {}".format(c)
def LF_common_2000(c):
    try:
        if(not type(c) == 'snorkel.models.candidate.BiomarkerCondition'):
            return
        if(c.condition.get_span() in common2000):
            return -1
    except:
        print "Threw out: {}".format(c)
#----------------------
# Biomarker-Drug (new)
#----------------------

def LF_colon(c):
    try:
        # if the word has a colon after it its generally not a biomarker
        words_in_between = []
        for thing in get_between_tokens(c):
            words_in_between.append(thing)
        if(len(words_in_between) > 1 and words_in_between[0] == ":"):
            return -1
    except:
        print "Threw out: {}".format(c)
def LF_roman_numeral(c):
    try:
        biomarker = (c.biomarker.get_span())
        unicodedata.normalize('NFKD', biomarker).encode('ascii', 'ignore')
        if re.match(r'((?<=\s)|(?<=^))(M{1,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})|M{0,4}(CM|CD|D?C{1,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})|M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{1,3})(IX|IV|V?I{0,3})|M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{1,3}))(?=\s)',
                    biomarker):
            return -1
    except:
        print "Threw out: {}".format(c)
def LF_same_thing_BD(c):
    try:
        if(not type(c) == 'snorkel.models.candidate.BiomarkerDrug'):
            return
        if(c.biomarker.get_span() == c.drug.get_span()):
            return -1
    except:
        print "Threw out: {}".format(c)
#------------------------
# Biomarker-Medium (new)
#------------------------

def LF_distance_far(c):
    try:
        x=0
        for thing in get_between_tokens(c):
            x+=1
        if x > 10:
            return -1
    except:
        print "Threw out: {}".format(c)

def LF_same_thing(c):
    try:
        if(not type(c) == 'snorkel.models.candidate.BiomarkerMedium'):
            return
        if(c.biomarker.get_span() == c.medium.get_span()):
            return -1
    except:
        print "Threw out: {}".format(c)

#------------------------
# Biomarker-Type (new)
#------------------------

def LF_same_thing(c):
    try:
        if(c.biomarker.get_span() == c.type.get_span()):
            return -1
    except:
        print "Threw out: {}".format(c)
    
    
    