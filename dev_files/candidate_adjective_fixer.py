from snorkel.lf_helpers import *
from snorkel.models import *
from random import randint


def fix_specificity(session, BiomarkerCondition, candidates):
    needsAdding = []
    justAdded = []

    for c in candidates:
        # Create the StableLabel for the adjective candidate
        temp_arr_poses = []
        temp_arr_words = []
        cared_words = []

        for thing in get_left_tokens(c.condition, window=4, attrib="words"):
            temp_arr_words.append(thing)
        for thing in get_left_tokens(c.condition, window=4, attrib="pos_tags"):
            temp_arr_poses.append(thing)

        x = (len(temp_arr_poses) - 1)
        while (x >= 0 and (temp_arr_poses[x] == 'jj' or temp_arr_poses[x] == 'nn')):
            cared_words.append(temp_arr_words[x])
            x -= 1

        to_append_string = ""
        for word in cared_words:
            to_append_string += (word + " ")
        print to_append_string + " added to " + str(c.condition.get_span())

        
def add_adj_candidate_BC(session, BiomarkerCondition, candidates, dSplit):
    needsAdding = []
    justAdded = []
    for c in candidates:
        temp_arr_poses = []
        temp_arr_words = []
        cared_words = []
        for thing in get_left_tokens(c.condition, window=4, attrib="words"):
            temp_arr_words.append(thing)
        for thing in get_left_tokens(c.condition, window=4, attrib="pos_tags"):
            temp_arr_poses.append(thing)
        x = (len(temp_arr_poses) - 1)
        while (x >= 0 and (temp_arr_poses[x] == 'jj' or temp_arr_poses[x] == 'nn')):
            cared_words.append(temp_arr_words[x])
            x -= 1

        to_append_string = ""
        for word in cared_words:
            to_append_string += (word + " ")

        '''If there are changes to be made'''
        if (len(to_append_string) > 0):

            '''The biomarker is the same'''
            BiomarkerSpan = c.biomarker
            temp_splits = c.condition.get_stable_id().split(":")
            NewCandiateStable = "%s::%s:%s:%s" % (temp_splits[0], "span", (c.condition.char_start - len(to_append_string)), c.condition.char_end)
            # ConditionSpan = Span(stable_id=c.condition.get_stable_id(),
            #                      sentence=c.condition.sentence,
            #                      char_start=(c.condition.char_start - len(to_append_string)),
            #                      char_end=c.condition.char_end)



            # NewCandidate = BiomarkerCondition(biomarker=BiomarkerSpan, condition=c.condition)
            # needsAdding.append(NewCandidate)


            '''Query for the Biomarker'''

            print "Checking Biomarker: " + str(BiomarkerSpan.get_stable_id())
            query = session.query(Span).filter(Span.stable_id == BiomarkerSpan.get_stable_id())

            if query.count() == 0:
                '''This means that the Biomarker was not in the database (shouldn't happen)'''
                print "WOAH ERROR"


            '''Query for the Condition'''

            print "Checking Condition: " + str(NewCandiateStable)
            query = session.query(Span).filter(Span.stable_id == NewCandiateStable)
            print query.count()
            ConditionSpan = Span(stable_id=NewCandiateStable,
                                 sentence_id=c.condition.sentence.id,
                                 char_start=(c.condition.char_start - len(to_append_string)),
                                 char_end=c.condition.char_end)
            if query.count() == 0:
                '''This means the condition could not be found'''
                print "Couldn't find: " + str(NewCandiateStable)

                session.add(ConditionSpan)
                # Possibly gets automatically added?
                # session.add()

                '''See if this relation already exists (it shouldn't)'''
            query = session.query(BiomarkerCondition).filter(BiomarkerCondition.biomarker_id == BiomarkerSpan.id)
            query = query.filter(BiomarkerCondition.condition_id == ConditionSpan.id)
            if query.count() == 0 and not BiomarkerSpan.id == None and not ConditionSpan.id == None:
                print 'ADDING NEW CANDIDATE'
                NewCandidate = (BiomarkerCondition(
                    biomarker_id=BiomarkerSpan.id,
                    condition_id=ConditionSpan.id,
                    split=dSplit
                ))
                session.add(NewCandidate)
                needsAdding.append(NewCandidate)

    print "Finished Processing Existing Candidates"
    for thing in needsAdding:
        if not thing == None:
            candidates.append(thing)


def add_adj_candidate_BD(session, BiomarkerDrug, candidates, dSplit):
    needsAdding = []
    justAdded = []
    for c in candidates:

        # Create the StableLabel for the adjective candidate
        temp_arr_poses = []
        temp_arr_words = []
        cared_words = []
        for thing in get_left_tokens(c.drug, window=4, attrib="words"):
            temp_arr_words.append(thing)
        for thing in get_left_tokens(c.drug, window=4, attrib="pos_tags"):
            temp_arr_poses.append(thing)
        x = (len(temp_arr_poses) - 1)
        while (x >= 0 and (temp_arr_poses[x] == 'jj' or temp_arr_poses[x] == 'nn')):
            cared_words.append(temp_arr_words[x])
            x -= 1

        to_append_string = ""
        for word in cared_words:
            to_append_string += (word + " ")
        print to_append_string + " added to " + str(c.drug.get_span())
        '''If there are changes to be made'''
        if (len(to_append_string) > 0):

            '''The biomarker is the same'''
            BiomarkerSpan = c.biomarker
            temp_splits = c.drug.get_stable_id().split(":")
            NewCandiateStable = "%s::%s:%s:%s" % (
                temp_splits[0], "span", (c.drug.char_start - len(to_append_string)), c.drug.char_end)
            '''Query for the Biomarker'''

            print "Checking Biomarker: " + str(BiomarkerSpan.get_stable_id())
            query = session.query(Span).filter(
                Span.stable_id == BiomarkerSpan.get_stable_id())

            if query.count() == 0:
                '''This means that the Biomarker was not in the database (shouldn't happen)'''
                print "WOAH ERROR"

            '''Query for the Condition'''

            print "Checking Condition: " + str(NewCandiateStable)
            query = session.query(Span).filter(
                Span.stable_id == NewCandiateStable)
            print query.count()
            ConditionSpan = Span(stable_id=NewCandiateStable,
                                 sentence_id=c.drug.sentence.id,
                                 char_start=(c.drug.char_start -
                                             len(to_append_string)),
                                 char_end=c.drug.char_end)
            if query.count() == 0:
                '''This means the condition could not be found'''
                print "Couldn't find: " + str(NewCandiateStable)

                session.add(ConditionSpan)
                # Possibly gets automatically added?
                # session.add()

                '''See if this relation already exists (it shouldn't)'''
            query = session.query(BiomarkerDrug).filter(
                BiomarkerDrug.biomarker_id == BiomarkerSpan.id)
            query = query.filter(BiomarkerDrug.drug == ConditionSpan.id)
            if query.count() == 0 and not BiomarkerSpan.id == None and not ConditionSpan.id == None:
                print 'ADDING NEW CANDIDATE'
                NewCandidate = (BiomarkerDrug(
                    biomarker_id=BiomarkerSpan.id,
                    drug_id=ConditionSpan.id,
                    split=dSplit
                ))
                session.add(NewCandidate)
                needsAdding.append(NewCandidate)

    print "Finished Processing Existing Candidates"
    for thing in needsAdding:
        if not thing == None:
            candidates.append(thing)


def add_adj_candidate_BM(session, BiomarkerMedium, candidates, dSplit):
    needsAdding = []
    justAdded = []
    for c in candidates:

        # Create the StableLabel for the adjective candidate
        temp_arr_poses = []
        temp_arr_words = []
        cared_words = []
        for thing in get_left_tokens(c.medium, window=4, attrib="words"):
            temp_arr_words.append(thing)
        for thing in get_left_tokens(c.medium, window=4, attrib="pos_tags"):
            temp_arr_poses.append(thing)
        x = (len(temp_arr_poses) - 1)
        while (x >= 0 and (temp_arr_poses[x] == 'jj' or temp_arr_poses[x] == 'nn')):
            cared_words.append(temp_arr_words[x])
            x -= 1

        to_append_string = ""
        for word in cared_words:
            to_append_string += (word + " ")
        print to_append_string + " added to " + str(c.medium.get_span())
        '''If there are changes to be made'''
        if (len(to_append_string) > 0):

            '''The biomarker is the same'''
            BiomarkerSpan = c.biomarker
            temp_splits = c.medium.get_stable_id().split(":")
            NewCandiateStable = "%s::%s:%s:%s" % (temp_splits[
                                                  0], "span", (c.medium.char_start - len(to_append_string)), c.medium.char_end)
            # ConditionSpan = Span(stable_id=c.condition.get_stable_id(),
            #                      sentence=c.condition.sentence,
            #                      char_start=(c.condition.char_start - len(to_append_string)),
            #                      char_end=c.condition.char_end)

            # NewCandidate = BiomarkerCondition(biomarker=BiomarkerSpan, condition=c.condition)
            # needsAdding.append(NewCandidate)

            '''Query for the Biomarker'''

            print "Checking Biomarker: " + str(BiomarkerSpan.get_stable_id())
            query = session.query(Span).filter(
                Span.stable_id == BiomarkerSpan.get_stable_id())

            if query.count() == 0:
                '''This means that the Biomarker was not in the database (shouldn't happen)'''
                print "WOAH ERROR"

            '''Query for the Condition'''

            print "Checking Condition: " + str(NewCandiateStable)
            query = session.query(Span).filter(
                Span.stable_id == NewCandiateStable)
            print query.count()
            ConditionSpan = Span(stable_id=NewCandiateStable,
                                 sentence_id=c.medium.sentence.id,
                                 char_start=(c.medium.char_start -
                                             len(to_append_string)),
                                 char_end=c.medium.char_end)
            if query.count() == 0:
                '''This means the condition could not be found'''
                print "Couldn't find: " + str(NewCandiateStable)

                session.add(ConditionSpan)
                # Possibly gets automatically added?
                # session.add()

                '''See if this relation already exists (it shouldn't)'''
            query = session.query(BiomarkerMedium).filter(
                BiomarkerMedium.biomarker_id == BiomarkerSpan.id)
            query = query.filter(BiomarkerMedium.medium == ConditionSpan.id)
            if query.count() == 0 and not BiomarkerSpan.id == None and not ConditionSpan.id == None:
                print 'ADDING NEW CANDIDATE'
                NewCandidate = (BiomarkerMedium(
                    biomarker_id=BiomarkerSpan.id,
                    medium_id=ConditionSpan.id,
                    split=dSplit
                ))
                session.add(NewCandidate)
                needsAdding.append(NewCandidate)

    print "Finished Processing Existing Candidates"
    for thing in needsAdding:
        if not thing == None:
            candidates.append(thing)
