# Markerville Backend
This the Github repository for the Stanford Canary Center's Markerville system's backend (MarkerSub/Petrel). Examples of machine output can be viewed in /outputs. 

## Files In This Repo
* /TROP2_outputs - contains data extracted from a corpus of 281 TROP2 full text papers.
	* TROP2_machine_processed_predictions_[entity]_output.csv - files containing the output of the model with evaluations of biomarker-entity relations.
	* TROP2_data.csv - amalgamation of the entity data
	* TROP2_positive_data.csv - exclusively the relations that were marked as positive relationships
	
* /articles - gold truth data and document ID mapping
* /databases - entity databases (used in conjunction with heuristics) for entity matchers
* /dev_files - experimental files and development scripts
* /outputs - extracted data for over 10,000 research papers (the entirety of PubMed Central's Open Access corpus articles beginning with 0-9A-B).
* /snorkel - Snorkel library files
* Biomarker[Entity]Training.ipynb - jupyter notebooks containing training mechanisms using Snorkel. Best example is BiomarkerConditionTraining.ipynb (these notebooks still require cleaning of comments and print statements).
* matchers.py - heuristics for entity matching
* LF.py - heuristics for unsupervised learning component of model
* pipeline.py - use of a trained model to evaluate extracted entities

### 1. Corpus Processing
* ##### Given that a model already exist for the desired entity
```python
from pipeline import *
corpus_parser = parse_corpus(XML_filepath)
run(candidate1, candidate2, relation_name, cand_1_ngrams, cand_2_ngrams, cand_1_matcher, cand_2_matcher, model_name, output_file_name, corpus_parser)
```
* XML_filepath
	* XML file formatted in the following manner:
```XML
<ArticleSet>
	<Article>
    	<article-id> PubmedID or other identifier </article-id>
        <text> Text of the article </text>
    </Article>
    ... (do this for every article to be processed)
</ArticleSet>
```
* candidate1 - desired candidate to extract (eg. biomarker)  
* candidate2 - desired candidate to extract (eg. condition)  
* relation_name - name of the relation
* cand_1_ngrams - ngrams to match for candidate1  
* cand_2_ngrams - ngrams to match for candidate2  
* cand_1_matcher - matcher for candidate1  
* cand_2_matcher - matcher for candidate2  
* model_name - name of the saved model (eg. drugs, mediums, etc.)  
* output_file_name - name of the output csv file (eg. output.csv)
* corpus_parser - created in line 2

For now this must be run in the Jupyter notebook via ./run.sh due to PATH configuration. 

* ##### Potential Issues in Processing
	* UTF-8 encoding issues (the XML file must be UTF-8 encoded)

### 2. Improving/Creating Models
##### 1. Candidate Extraction  
###### 2.1.a. Corpus Parsing

The documents are initially passed into the Snorkel CorpusParser which separates the articles into sentences. 

###### 2.1.b. Abbreviation Matcher (missing?)

Often the abbreviations specific to a paper are defined at the beginning of the paper or when the full form is first mentioned. This information is grabbed by the abbreviation matcher and then added to the matchers. This also has potential to be used to build up a comprehensive abbreviation database. As of now, this code is not implemented in the pipeline, but the bulk of the code necessary for this process is available. 

###### 2.1.c. Candidate Matching

The matcher functions used to match specific entities are found in the matchers.py file. These matchers are Snorkel Matcher objects and generally rely on regular expressions and dictionaries for matching. If the goal is to improve recall, these matchers should be modified. 

###### 2.1.d. Adjective Grabbing

While regular expressions and dictionaries are able to grab disease names reliably, more specific information, such as disease stage and severity is extracted using the adjective grabber. 

##### 2. Improvement of Model Accuracy
###### 2.2.a. Labelling Functions

This is the only way to modify the accuracy of the model. By building better labelling functions or modifying older ones, the model accuracy will improve. When creating the labelling functions, it is important to include positive labelling functions as well. In the past the fully trained model would only return negative predictions (claim every relation was negative) because there were no positive labelling functions present. 

###### 2.2.b. Train, Dev, Test Sets

The current train, dev, and test sets consist only of abstracts rather than full text; however, this model is being used on full text. Thus, compiling train, dev, and test sets with full text rather than abstracts will likely improve performance. 

##### 3. Evaluation of Model Accuracy

###### 2.3.a. Annotation of Data

This is the most time consuming and currently the greatest issue with the pipeline. A properly annotated data set does not exist to use as train, test, and dev.  

Annotations should be at a **sentence** level. The pipeline extracts data at a sentence level, and thus Document level annotations will not work unless Snorkel's built in evaluation system is bypassed, which is not recommended. 

Another issue what we faced was with specificity of the annotations. For example, the annotation may pull this relation

```
TROP2 > castration resistant prostate cancer
```

Whereas the pipeline extracts: 

```
TROP2 > prostate cancer
```

or sometimes

```
TROP2 > cancer
```

This information extracted by the pipeline is not technically incorrect; however, it is not as specific as the annotation, so it is marked as incorrect. A determination needs to be made regarding whether  less specific mentions should be accepted. A potential solution is to annotate both the highly specific and the unspecific relations so that the accuracy measurement will remain accurate, but the recall measurement will reflect the lack of specificity. 

A corpus of full texts must be annotated at a sentence level in order for there to be better evaluation and more accurate hyperparamater searches. 





