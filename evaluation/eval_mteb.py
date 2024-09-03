import argparse
import os
from functools import partial

from mteb import MTEB
import torch

from gritlm import GritLM

SET_TO_TASK_TO_DS_TO_PROMPT = {
    # https://github.com/microsoft/unilm/blob/16da2f193b9c1dab0a692c6e4380bd43e70a40cd/e5/utils.py#L93
    'e5': {
        "Classification": {
            'AmazonCounterfactualClassification': 'Classify a given Amazon customer review text as either counterfactual or not-counterfactual',
            'AmazonPolarityClassification': 'Classify Amazon reviews into positive or negative sentiment',
            'AmazonReviewsClassification': 'Classify the given Amazon review into its appropriate rating category',
            'Banking77Classification': 'Given a online banking query, find the corresponding intents',
            'EmotionClassification': 'Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise',
            'ImdbClassification': 'Classify the sentiment expressed in the given movie review text from the IMDB dataset',
            'MassiveIntentClassification': 'Given a user utterance as query, find the user intents',
            'MassiveScenarioClassification': 'Given a user utterance as query, find the user scenarios',
            'MTOPDomainClassification': 'Classify the intent domain of the given utterance in task-oriented conversation',
            'MTOPIntentClassification': 'Classify the intent of the given utterance in task-oriented conversation',
            'ToxicConversationsClassification': 'Classify the given comments as either toxic or not toxic',
            'TweetSentimentExtractionClassification': 'Classify the sentiment of a given tweet as either positive, negative, or neutral',
        },
        "Clustering": {
            'ArxivClusteringP2P': 'Identify the main and secondary category of Arxiv papers based on the titles and abstracts',
            'ArxivClusteringS2S': 'Identify the main and secondary category of Arxiv papers based on the titles',
            'BiorxivClusteringP2P': 'Identify the main category of Biorxiv papers based on the titles and abstracts',
            'BiorxivClusteringS2S': 'Identify the main category of Biorxiv papers based on the titles',
            'MedrxivClusteringP2P': 'Identify the main category of Medrxiv papers based on the titles and abstracts',
            'MedrxivClusteringS2S': 'Identify the main category of Medrxiv papers based on the titles',
            'RedditClustering': 'Identify the topic or theme of Reddit posts based on the titles',
            'RedditClusteringP2P': 'Identify the topic or theme of Reddit posts based on the titles and posts',
            'StackExchangeClustering': 'Identify the topic or theme of StackExchange posts based on the titles',
            'StackExchangeClusteringP2P': 'Identify the topic or theme of StackExchange posts based on the given paragraphs',
            'TwentyNewsgroupsClustering': 'Identify the topic or theme of the given news articles',
        },
        "PairClassification": {
            'SprintDuplicateQuestions': 'Retrieve duplicate questions from Sprint forum',
            'TwitterSemEval2015': 'Retrieve tweets that are semantically similar to the given tweet',
            'TwitterURLCorpus': 'Retrieve tweets that are semantically similar to the given tweet',
        },
        "Reranking": {
            'AskUbuntuDupQuestions': {
                'query': 'Retrieve duplicate questions from AskUbuntu forum',
                'corpus': 'Retrieve duplicate questions from AskUbuntu forum',
            },
            'MindSmallReranking': {
                'query': 'Retrieve relevant news articles based on user browsing history',
                'corpus': 'Retrieve relevant news articles based on user browsing history',
            },
            'SciDocsRR': {
                'query': 'Given a title of a scientific paper, retrieve the titles of other relevant papers',
                'corpus': 'Given a title of a scientific paper, retrieve the titles of other relevant papers',
            },
            'StackOverflowDupQuestions': {
                'query': 'Retrieve duplicate questions from StackOverflow forum',
                'corpus': 'Retrieve duplicate questions from StackOverflow forum',
            },
        },
        'Retrieval': {
            'ArguAna': {
                'query': 'Given a claim, find documents that refute the claim',
                'corpus': '',
            },
            'ClimateFEVER': {
                'query': 'Given a claim about climate change, retrieve documents that support or refute the claim',
                'corpus': '',
            },
            'CQADupstackRetrieval': {
                'query': 'Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question',
                'corpus': '',
            },
            'DBPedia': {
                'query': 'Given a query, retrieve relevant entity descriptions from DBPedia',
                'corpus': '',
            },
            'FEVER': {
                'query': 'Given a claim, retrieve documents that support or refute the claim',
                'corpus': '',
            },
            'FiQA2018': {
                'query': 'Given a financial question, retrieve user replies that best answer the question',
                'corpus': '',
            },
            'HotpotQA': {
                'query': 'Given a multi-hop question, retrieve documents that can help answer the question',
                'corpus': '',
            },
            'MSMARCO': {
                'query': 'Given a web search query, retrieve relevant passages that answer the query',
                'corpus': '',
            },
            'NFCorpus': {
                'query': 'Given a question, retrieve relevant documents that best answer the question',
                'corpus': '',
            },
            'NQ': {
                'query': 'Given a question, retrieve Wikipedia passages that answer the question',
                'corpus': '',
            },
            'QuoraRetrieval': {
                'query': 'Given a question, retrieve questions that are semantically equivalent to the given question',
                'corpus': '',
            },
            'SCIDOCS': {
                'query': 'Given a scientific paper title, retrieve paper abstracts that are cited by the given paper',
                'corpus': '',
            },
            'SciFact': {
                'query': 'Given a scientific claim, retrieve documents that support or refute the claim',
                'corpus': '',
            },
            'Touche2020': {
                'query': 'Given a question, retrieve detailed and persuasive arguments that answer the question',
                'corpus': '',
            },
            'TRECCOVID': {
                'query': 'Given a query on COVID-19, retrieve documents that answer the query',
                'corpus': '',
            },
        },
        'STS': {
            'STS12': "Retrieve semantically similar text.",
            'STS13': "Retrieve semantically similar text.",
            'STS14': "Retrieve semantically similar text.",
            'STS15': "Retrieve semantically similar text.",
            'STS16': "Retrieve semantically similar text.",
            'STS17': "Retrieve semantically similar text.",
            'STS22': "Retrieve semantically similar text.",
            'BIOSSES': "Retrieve semantically similar text.",
            'SICK-R': "Retrieve semantically similar text.",
            'STSBenchmark': "Retrieve semantically similar text.",
        },            
        'Summarization': {
            'SummEval': "Given a news summary, retrieve other semantically similar summaries",
        },
    },
    'medi2': {
        "Classification": {
            'Banking77Classification': 'Represent the text for finding another one-sentence banking query with the same intent',
            'EmotionClassification': 'Represent the text for finding another one-sentence text with the same emotion',
            'AmazonCounterfactualClassification': 'Represent the text to find another sentence with the same counterfactuality, e.g. sentences with "would", "wish", etc. should match with other sentences of that kind.',
            'ImdbClassification': 'Represent the text for finding another one-sentence movie review with the same sentiment',
            'MassiveIntentClassification': 'Represent the text for finding another text of a few words with the same intent',
            'MassiveScenarioClassification': 'Represent the text for finding another text of a few words about the same scenario',
            'MTOPDomainClassification': 'Represent the text for finding another text of a few words about the same domain',
            'MTOPIntentClassification': 'Represent the text for finding another text of a few words with the same intent',
            'ToxicConversationsClassification': 'Represent the text for finding another comment of up to a passage in length with the same level of toxicity (either toxic or not toxic)',
            'AmazonPolarityClassification': 'Represent the review for finding another Amazon review with the same sentiment (positive / negative)',
            'AmazonReviewsClassification': 'Represent the review for finding another Amazon review with the same rating',
            'TweetSentimentExtractionClassification': 'Represent the tweet for finding another tweet with the same sentiment (positive / neutral / negative)',
        },
        "Clustering": {
            'TwentyNewsgroupsClustering': 'Represent the title to find a similar news title from the same newsgroup',
            'BiorxivClusteringS2S': 'Represent the text to find another bioRxiv title about the same topic',
            'BiorxivClusteringP2P': 'Represent the text to find another bioRxiv title with abstract (concatenated) about the same topic',
            'ArxivClusteringS2S': 'Represent the text to find another arXiv title about the same topic',
            'ArxivClusteringP2P': 'Represent the text to find another arXiv title with abstract (concatenated) about the same topic',
            'MedrxivClusteringS2S': 'Represent the text to find another medRxiv title about the same topic',
            'MedrxivClusteringP2P': 'Represent the text to find another medRxiv title with abstract (concatenated) about the same topic',
            'RedditClustering': 'Represent the text to find another Reddit community title that stems from the same subreddit',
            'RedditClusteringP2P': 'Represent the text to find another Reddit community title with post (concatenated) from the same subreddit',
            'StackExchangeClustering': 'Represent the text to find another StackExchange title that stems from the same StackExchange',
            'StackExchangeClusteringP2P': 'Represent the text to find another StackExchange title with post (concatenated) that stems from the same StackExchange',
        },
        "PairClassification": {
            'SprintDuplicateQuestions': 'Represent the question to be matched with another duplicate user question from the Sprint community forum',
            'TwitterSemEval2015': 'Represent the tweet to find another tweet that is a paraphrase of it',
            'TwitterURLCorpus': 'Represent the tweet to find another tweet that is a paraphrase of it',
        },
        'Reranking': {
            # Questions from AskUbuntu with manual annotations marking pairs of questions as similar or dissimilar.
            'AskUbuntuDupQuestions': {
                'query': 'Represent the query to find a duplicate query on the AskUbuntu community forum',
                'corpus': 'Represent the query to find a duplicate query on the AskUbuntu community forum',
            },
            # Stack Overflow Duplicate Questions Task for questions with the tags Java, JavaScript and Python, ranking questions as duplicates or not
            'StackOverflowDupQuestions': {
                'query': 'Represent the query to find a duplicate query on the StackOverflow Java/JavaScript/Python community forums',
                'corpus': 'Represent the query to find a duplicate query on the StackOverflow Java/JavaScript/Python community forums',
            },
            # Both are titles, e.g. "Beauty eMakeup: A Deep Makeup Transfer System" matches with "Makeup like a superstar: Deep Localized Makeup Transfer Network"
            'SciDocsRR': {
                'query': 'Represent the title to find a similar scientific paper title',
                'corpus': 'Represent the title to find a similar scientific paper title',
            },
            # E.g. "Taylor Swift Says Scooter Braun, Scott Borchetta Are Blocking Her From Playing Old Hits at AMAs" matches with "Author Jennine Capó Crucet responds after white college students burn her book" but not with "How to Make Buttermilk At Home With Just 2 Ingredients"
            'MindSmallReranking': {
                'query': 'Represent the news headline to find another news headline that the same reader would enjoy',
                'corpus': 'Represent the news headline to find another news headline that the same reader would enjoy',
            },
        },
        'Retrieval': {
            ### Bio-Medical Information Retrieval ###
            # NFCorpus [7] contains natural language queries harvested from NutritionFacts (NF). We use the original splits provided alongside all content sources from NF (videos, blogs, and Q&A posts) as queries Q and annotated medical documents from PubMed as corpus T.
            'NFCorpus': {
                'query': 'Represent the query from NutritionFacts to find a title with text of a medical document from PubMed about it',
                'corpus': 'Represent this text of a medical document from PubMed to find a query someone may enter at NutritionFacts that it answers',
            },
            # TREC-COVID [65] is an ad-hoc search challenge based on the CORD-19 dataset containing scientific articles related to the COVID-19 pandemic [69]. We include the July 16, 2020 version of CORD-19 dataset as corpus T and use the final cumulative judgements with query descriptions from the original task as queries Q.
            'TRECCOVID': {
                'query': 'Represent the search query to find a scientific article about COVID-19 that adequately addresses the query',
                'corpus': 'Represent the scientific article about COVID-19 to find a user query that it adequately addresses'
            },
            ### Open-domain Question Answering (QA) ###
            'MSMARCO': {
                'query': 'Represent the Bing user search query to find a passage that adequately addresses it',
                'corpus': 'Represent the passage for finding a Bing user search query about it',
            },
            # Natural Questions [34] contains Google search queries and documents with paragraphs and answer spans within Wikipedia articles. We did not use the NQ version from ReQA [1] as it focused on queries having a short answer. As a result, we parsed the HTML of the original NQ dataset and include more complex development queries that often require a longer passage as answer compared to ReQA. We filtered out queries without an answer, or having a table as an answer, or with conflicting Wikipedia pages. We retain 2,681,468 passages as our corpus T and 3452 test queries Q from the original dataset.
            'NQ': {
                'query': 'Represent the Google search query to find an answer span from a Wikipedia article that addresses it',
                'corpus': 'Represent the Wikipedia article span to find a Google search query that would be addressed by it',
            },
            # HotpotQA [76] contains multi-hop like questions which require reasoning over multiple paragraphs to find the correct answer. We include the original full-wiki task setting: utilizing processed Wikipedia passages as corpus T. We held out randomly sampled 5447 queries from training as our dev split. We use the original (paper) task’s development split as our test split Q.
            'HotpotQA': {
                # Wikipedia Question
                'query': 'Represent the multi-hop question to find a Wikipedia passage that answers it',
                # Wikipedia Articles
                'corpus': 'Represent the Wikipedia passage to find a multi-hop question that it answers',
            },
            # FiQA-2018 [44] Task 2 consists of opinion-based question-answering. We include financial data by crawling StackExchange posts under the Investment topic from 2009-2017 as our corpus T. We randomly sample out 500 and 648 queries Q from the original training split as dev and test splits.            
            'FiQA2018': {
                'query': 'Represent the StackExchange user query to find a StackExchange post from the Investment topic that answers it',
                'corpus': 'Represent the StackExchange post from the Investment topic to find a StackExchange user query that it answers',
            },
            ### Argument Retrieval ###
            # ArguAna Counterargs Corpus [67] involves the task of retrieval of the best counterargument to an argument. We include pairs of arguments and counterarguments scraped from the online debate portal as corpus T. We consider the arguments present in the original test split as our queries Q.            
            'ArguAna': {
                'query': 'Represent the passage to find a passage with a counter-argument about the same topic to it',
                'corpus': 'Represent the passage to find a passage with a counter-argument about the same topic to it',
            },
            # Touché-2020 [6] Task 1 is a conversational argument retrieval task. We use the conclusion as title and premise for arguments present in args.me [66] as corpus T. We include the shared Touché-2020 task data as our test queries Q. The original relevance judgements (qrels) file also included negative judgements (-2) for non-arguments present within the corpus, but for simplicity we substitute them as zero.            
            'Touche2020': {
                'query': 'Represent the question to find a title with passage of an argument from args.me that takes a stance about it',
                'corpus': 'Represent the title with passage of an argument from args.me to find a question that it takes a stance about',
            },
            ### Duplicate Question Retrieval ###
            # CQADupStack [25] is a popular dataset for research in community question-answering (cQA). The corpus T comprises of queries from 12 different StackExchange subforums: Android, English,Gaming, Gis, Mathematica, Physics, Programmers, Stats, Tex, Unix, Webmasters and Wordpress. We utilize the original test split for our queries Q, and the task involves retrieving duplicate query (title + body) for an input query title. We evaluate each StackExchange subforum separately and report the overall mean scores for all tasks in BEIR.            
            # Example query: Android chroot ubuntu - is it possible to get ubuntu to recognise usb devices
            # Example doc: I want to send files to android tablet with a application from PC. - I can send files directly to tablet (2.3 android OS) PC see it as a external usb drive. - But i can't send files to tablet (4.2 android OS), because PC see it as a portable media player.(MTP) - How can i fix this problem ? - How can show my device as a external drive? my application that sent files written via Delphi.
            # Example doc title: How can show android tablet as a external storage to PC?
            'CQADupstackTexRetrieval': {
                'query': 'Represent the title of a user question to find a duplicate user question title with body from the Tex StackExchange forum',
                'corpus': 'Represent the question title with body posted by a user to find a duplicate user question title from the Tex StackExchange forum',
            },
            'CQADupstackWebmastersRetrieval': {
                'query': 'Represent the title of a user question to find a duplicate user question title with body from the Webmasters StackExchange forum',
                'corpus': 'Represent the question title with body posted by a user to find a duplicate user question title from the Webmasters StackExchange forum',
            },
            'CQADupstackEnglishRetrieval': {
                'query': 'Represent the title of a user question to find a duplicate user question title with body from the English StackExchange forum',
                'corpus': 'Represent the question title with body posted by a user to find a duplicate user question title from the English StackExchange forum',
            },
            'CQADupstackGamingRetrieval': {
                'query': 'Represent the title of a user question to find a duplicate user question title with body from the Gaming StackExchange forum',
                'corpus': 'Represent the question title with body posted by a user to find a duplicate user question title from the Gaming StackExchange forum',
            },
            'CQADupstackGisRetrieval': {
                'query': 'Represent the title of a user question to find a duplicate user question title with body from the Gis StackExchange forum',
                'corpus': 'Represent the question title with body posted by a user to find a duplicate user question title from the Gis StackExchange forum',
            },
            'CQADupstackUnixRetrieval': {
                'query': 'Represent the title of a user question to find a duplicate user question title with body from the Unix StackExchange forum',
                'corpus': 'Represent the question title with body posted by a user to find a duplicate user question title from the Unix StackExchange forum',
            },
            'CQADupstackMathematicaRetrieval': {
                'query': 'Represent the title of a user question to find a duplicate user question title with body from the Mathematica StackExchange forum',
                'corpus': 'Represent the question title with body posted by a user to find a duplicate user question title from the Mathematica StackExchange forum',
            },
            'CQADupstackStatsRetrieval': {
                'query': 'Represent the title of a user question to find a duplicate user question title with body from the Stats StackExchange forum',
                'corpus': 'Represent the question title with body posted by a user to find a duplicate user question title from the Stats StackExchange forum',
            },
            'CQADupstackPhysicsRetrieval': {
                'query': 'Represent the title of a user question to find a duplicate user question title with body from the Physics StackExchange forum',
                'corpus': 'Represent the question title with body posted by a user to find a duplicate user question title from the Physics StackExchange forum',
            },
            'CQADupstackProgrammersRetrieval': {
                'query': 'Represent the title of a user question to find a duplicate user question title with body from the Programmers StackExchange forum',
                'corpus': 'Represent the question title with body posted by a user to find a duplicate user question title from the Programmers StackExchange forum',
            },
            'CQADupstackAndroidRetrieval': {
                'query': 'Represent the title of a user question to find a duplicate user question title with body from the Android StackExchange forum',
                'corpus': 'Represent the question title with body posted by a user to find a duplicate user question title from the Android StackExchange forum',
            },
            'CQADupstackWordpressRetrieval': {
                'query': 'Represent the title of a user question to find a duplicate user question title with body from the Wordpress StackExchange forum',
                'corpus': 'Represent the question title with body posted by a user to find a duplicate user question title from the Wordpress StackExchange forum',
            },
            # Quora Duplicate Questions dataset identifies whether two questions are duplicates. Quora originally released containing 404,290 question pairs. We add transitive closures to the original dataset. Further, we split it into train, dev, and test sets with a ratio of about 85%, 5% and 10% of the original pairs. We remove all overlaps between the splits and ensure that a question in one split of the dataset does not appear in any other split to mitigate the transductive classification problem [27]. We achieve 522,931 unique queries as our corpus T and 5,000 dev and 10,000 test queries Q respectively
            'QuoraRetrieval': {
                'query': 'Represent the Quora question to find another short duplicate question on Quora',
                'corpus': 'Represent the Quora question to find another short duplicate question on Quora',
            },
            ### Entity Retrieval ###
            # DBPedia-Entity-v2 [21] is an established entity retrieval dataset. It contains a set of heterogeneous entity-bearing queries Q containing named entities, IR style keywords, and natural language queries. The task involves retrieving entities from the English part of DBpedia corpus T from October 2015. We randomly sample out 67 queries from the test split as our dev set.
            'DBPedia': {
                'query': 'Represent the entity to find a title with abstract about this entity from the DBPedia corpus',
                'corpus': 'Represent the title with abstract of a DBPedia corpus entry to find the entity of a few words it is about',
            },
            ### Citation Prediction ###
            # SCIDOCS [9] contains a corpus T of 30K held-out pool of scientific papers. We consider the direct-citations (1 out of 7 tasks mentioned in the original paper) as the best suited task for retrieval evaluation in BEIR. The task includes 1k papers as queries Q with 5 relevant papers and 25 (randomly selected) uncited papers for each query.
            'SCIDOCS': {
                'query': 'Represent the scientific paper title to find the title with abstract of a scientific paper on PubMed that it has likely cited',
                'corpus': 'Represent the title with abstract of this scientific paper to find the title of another scientific paper on PubMed that likely cites this article',
            },
            ### Fact Checking ###
            # FEVER [60] The Fact Extraction and VERification dataset is collected to facilitate the automatic fact checking. We utilize the original paper splits as queries Q and retrieve evidences from the pre-processed Wikipedia Abstracts (June 2017 dump) as our corpus T.
            'FEVER': {
                'query': 'Represent the claim to find a Wikipedia abstract to support it',
                # Wikipedia Articles
                'corpus': 'Represent the Wikipedia abstract to find a claim that it supports',
            },
            'ClimateFEVER': {
                # Climate-based Claim
                'query': 'Represent the climate-based claim to find a Wikipedia abstract to support it',
                # Wikipedia Articles
                'corpus': 'Represent the Wikipedia abstract to find a climate-related claim that it supports',
            },
            # SciFact [68] verifies scientific claims using evidence from the research literature containing scientific paper abstracts. We use the original publicly available dev split from the task containing 300 queries as our test queries Q, and include all documents from the original dataset as our corpus T.
            'SciFact': {
                'query': 'Represent the scientific claim to find a scientific paper abstract from PubMed to support it',
                'corpus': 'Represent the scientific paper abstract from PubMed to find a scientific claim that it supports',
            },
        },
        'STS': {
            # Other prompt candidates:
            # 'Represent the sentence to find another single-sentence casual post about the same topic',
            'STS12': 'Represent the sentence to find another sentence with the same meaning',
            'STS13': 'Represent the sentence to find another sentence with the same meaning',
            # For the English subtask, we exposed the systems to a diversity of testing scenarios, by preparing additional OntoNotesWordNet sense mappings and news headlines, as well as introducing new genres, including image descriptions, DEFT discussion forums, DEFT newswire, and tweet-newswire headline mappings
            'STS14': 'Represent the sentence to find another sentence with the same meaning',
            'STS15': 'Represent the sentence to find another sentence with the same meaning',
            'STS16': 'Represent the sentence to find another sentence with the same meaning',
            'STS17': 'Represent the sentence to find another sentence with the same meaning',
            'STS22': 'Represent the sentence to find another sentence with the same meaning',
            'BIOSSES': 'Represent the text to find another biological statement with the same meaning',
            # Sentences Involving Compositional Knowledge (SICK) contains a large number of sentence pairs (10 0000) that are lexically, syntactically and semantically rich.
            'SICK-R': 'Represent the sentence to find another sentence with the same meaning',
            'STSBenchmark': 'Represent the sentence to find another sentence with the same meaning',
        },
        'Summarization': {
            'SummEval': {
                'query': 'Represent the human-written summary to find a high-quality machine-written summary of the same news article',
                'corpus': 'Represent the machine-written summary to find a human-written summary with similar quality of the same news article',
            },
        },        
    },
    'instructor-xl': {
        "Classification": {
            'Banking77Classification': 'Represent the bank77 purposes for retrieving its bank intent: ',
            'EmotionClassification':  'Represent the amazon emotion sentence for classifying the emotion: ',
            'AmazonCounterfactualClassification': 'Represent Daily casual counter-sentences for categorization as correct-sentences or counter-sentences: ',
            'ImdbClassification': 'Represent a review sentence for classifying emotion as positive or negative: ',
            'MassiveIntentClassification':'Represent the sentence for categorizing its task intent as qa_maths, takeaway_order, audio_volume_other, recommendation_movies, iot_cleaning, qa_stock, or recommendation_locations: ',
            'MassiveScenarioClassification': "represent an ms sentence for retrieving its intent: ",
            'MTOPDomainClassification': 'represent a MTO sentence to retrieve the task intention: ',
            'MTOPIntentClassification': 'Represent an mto sentence for retrieving its behind task intention: ',
            'ToxicConversationsClassification': 'Represent a toxicity comment for classifying its toxicity as toxic or non-toxic: ',
            'AmazonPolarityClassification': 'Represent the sentiment comment for retrieving a duplicate sentence: ',
            'AmazonReviewsClassification': 'Represent an amazon review sentence to find the emoation; ',
            'TweetSentimentExtractionClassification': 'Represent Daily-life spoken sentences for categorization; Input: ',
        },
        "Clustering": {
            'TwentyNewsgroupsClustering': 'Represent the news comment for clustering; ',
            'BiorxivClusteringS2S': 'Represent the biological statement for retrieval; ',
            'MedrxivClusteringS2S': 'Represent the Biological statement for clustering biological statements: ',
            'ArxivClusteringP2P': 'Represent the science passage for retrieval: ',
            'ArxivClusteringS2S': 'Represent the Science statements for retrieval: ',
            'BiorxivClusteringP2P': 'Represent the Biological passage for retrieval: ',
            'MedrxivClusteringP2P': 'Represent the Biological paragraph for retrieval: ',
            'RedditClustering': 'represent the Reddit community title: ',
            'RedditClusteringP2P': 'represent a Reddit community passage: ',
            'StackExchangeClustering': 'Represent a question for retrieval: ',
            'StackExchangeClusteringP2P': 'Represent the question and answer passage for retrieving relevant question and answer passages: ',
        },
        "PairClassification": {
            'TwitterSemEval2015': 'Represent the twitter post for retrieving comments: ',
            'TwitterURLCorpus': 'represent a Twitter posts for retrieval: ',
            'SprintDuplicateQuestions': 'represent the Sprint questions for retrieving relevant posts, ',
        },
        'Reranking': {
            'AskUbuntuDupQuestions': {
                'query': 'Represent the Ubuntu question to retrieve a duplicate question: ',
                'corpus': 'Represent the Ubuntu question: ',
            },
            'StackOverflowDupQuestions': {
                'query': 'Represent the StackOverflow question: ',
                'corpus': 'Represent the StackOverflow question: ',
            },
            'SciDocsRR': {
                'query': 'Represent the Science question: ',
                'corpus': 'Represent the Science document: '
            },
            'MindSmallReranking': {
                'query': 'Represent the news query for retrieving articles: ',
                'corpus': 'Represent the news article for retrieval: ',
            },
        },
        'Retrieval': {
            'ClimateFEVER': {
                'query': 'Represent the Climate question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
            'HotpotQA': {
                'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
            'FEVER': {
                'query': 'Represent the fact for retrieving supporting evidence: ',
                'corpus': 'Represent the evidence for retrieval: ',
            },
            'MSMARCO': {
                'query': 'Represent the question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
            'DBPedia': {
                'query': 'Represent the Wikipedia questions to retrieve a supporting document: ',
                'corpus': 'Represent the Wikipedia documents for retrieval: ',
            },
            'NQ': {
                'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
            'QuoraRetrieval': {
                'query': 'Represent the Quora question to retrieve question: ',
                'corpus': 'Represent the Quora question to retrieve question: ',
            },
            'SCIDOCS': {
                'query': 'Represent a Science question for retrieving supporting papers: ',
                'corpus': 'Represent the Science paper: ',
            },
            'TRECCOVID': {
                'query': 'Represent the Coronavirus questions to retrieve a supporting document: ',
                'corpus': 'Represent the Coronavirus documents for retrieval: ',
            },
            'Touche2020': {
                'query': 'Represent questions: ',
                'corpus': 'Represent arguments: ',
            },
            'SciFact': {
                'query': 'Represent the Scientific queries for retrieving a supporting passage: ',
                'corpus': 'represent the scientific paragraph for retrieval: ',
            },
            'NFCorpus': {
                'query': 'Represent the nutrition facts to retrieve Public medical articles: ',
                'corpus': 'Represent the Public medical articles for retrieval: ',
            },
            'ArguAna': {
                'query': 'Represent Debating conversations to retrieve a counter-argument: ',
                'corpus': 'Represent counter-arguments: ',
            },
            'CQADupstackTexRetrieval': {
                'query': 'Represent the question for retrieving answers: ',
                'corpus': 'Represent the answer for retrieval: ',
            },
            'CQADupstackWebmastersRetrieval': {
                'query': 'Represent the Webmaster question for retrieving answers: ',
                'corpus': 'Represent the Webmaster answer: ',
            },
            'CQADupstackEnglishRetrieval': {
                'query': 'Represent the English question for retrieving documents: ',
                'corpus': 'Represent the English answer for retrieval: ',
            },
            'CQADupstackGamingRetrieval': {
                'query': 'Represent the Gaming question for retrieving answers: ',
                'corpus': 'Represent the Gaming answer for retrieval: ',
            },
            'CQADupstackGisRetrieval': {
                'query': 'Represent the Gis question for retrieving answers: ',
                'corpus': 'Represent the Gis answer for retrieval: ',
            },
            'CQADupstackUnixRetrieval': {
                'query': 'Represent the Unix questions to retrieve a supporting answer: ',
                'corpus': 'Represent the Unix answers for retrieval: ',
            },
            'CQADupstackMathematicaRetrieval': {
                'query': 'Represent the Mathematical question for retrieving answers: ',
                'corpus': 'Represent the Mathematical answer for retrieval: ',
            },
            'CQADupstackStatsRetrieval': {
                'query': 'Represent the Statistical question for retrieving answers: ',
                'corpus': 'Represent the Statistical answer for retrieval: ',
            },
            'CQADupstackPhysicsRetrieval': {
                'query': 'Represent the Physics question for retrieving answers: ',
                'corpus': 'Represent the Physics answer for retrieval: ',
            },
            'CQADupstackProgrammersRetrieval': {
                'query': 'Represent the Programming question for retrieving answers: ',
                'corpus': 'Represent the Programming answer for retrieval: ',
            },
            'CQADupstackAndroidRetrieval': {
                'query': 'Represent the Android question for retrieving answers: ',
                'corpus': 'Represent the Android answer for retrieval: ',
            },
            'CQADupstackWordpressRetrieval': {
                'query': 'Represent the Wordpress question for retrieving answers: ',
                'corpus': 'Represent the Wordpress answer for retrieval: ',
            },
            'FiQA2018': {
                'query': 'Represent the finance questions to retrieve a supporting answer: ',
                'corpus': 'Represent the finance answers for retrieval: ',
            },
        },        
        'STS': {
            'STS12': 'represent texts, ',
            'STS13': 'represent a casual post, ',
            'STS14': 'Represent a post; ',
            'STS15': 'Represent a posts,,, ',
            'STS16': 'Represent posts: ',
            'STS17': 'Represent a statement, ',
            'STS22': 'represent the statement: ',
            'BIOSSES': 'represent the Biological statement: ',
            'SICK-R': 'Represent a post: ',
            'STSBenchmark': 'represent posts, ',
        },
        'Summarization': {
            'SummEval': 'Represent the news statement for retrieval: ',
        },
    },
    'instructor-base': {
        'STS': {
            'STS12': 'Represent the statement, ',
            'STS13': 'represent the statement, ',
            'STS14': 'Represent the statement, ',
            'STS15': 'Represent the post, ',
            'STS16': 'Represent the post: ',
            'STS17': 'Represent the sentence for classification: ',
            'STS22': 'represent the statement: ',
            'BIOSSES': 'Represent the Bio-medical statement: ',
            'SICK-R': 'Represent the statement: ',
            'STSBenchmark': 'represent the statement: ',
        },
        'Retrieval': {
            'ClimateFEVER': {
                'query': 'Represent the Climate question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
            'HotpotQA': {
                'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
            'FEVER': {
                'query': 'Represent the fact for retrieving supporting evidence: ',
                'corpus': 'Represent the evidence for retrieval: ',
            },
            'MSMARCO': {
                'query': 'Represent the question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
            'DBPedia': {
                'query': 'Represent the Wikipedia sentence for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
            'NQ': {
                'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
            'QuoraRetrieval': {
                'query': 'Represent the Quora question for retrieving duplicate questions: ',
                'corpus': 'Represent the Quora question for retrieving duplicate questions: ',
            },
            'SCIDOCS': {
                'query': 'Represent a Science question for retrieving supporting papers: ',
                'corpus': 'Represent the Science paper: ',
            },
            'TRECCOVID': {
                'query': 'Represent the Coronavirus question for retrieving supporting documents: ',
                'corpus': 'Represent the Coronavirus document for retrieval: ',
            },
            'Touche2020': {
                'query': 'Represent a question: ',
                'corpus': 'Represent an argument: ',
            },
            'SciFact': {
                'query': 'Represent a Scientific query for retrieving a supporting passage; ',
                'corpus': 'represent the Scientific passage for retrieval; ',
            },
            'NFCorpus': {
                'query': 'Represent the Medicine question for retrieving a relevant document: ',
                'corpus': 'Represent the medical document for retrieval: ',
            },
            'ArguAna': {
                'query': 'Represent the Debate argument for retrieving a counter-argument: ',
                'corpus': 'Represent the Counter debate argument: ',
            },
            'CQADupstackTexRetrieval': {
                'query': 'Represent the question for retrieving answers: ',
                'corpus': 'Represent the answer for retrieval: ',
            },
            'CQADupstackWebmastersRetrieval': {
                'query': 'Represent the Webmaster question for retrieving answers: ',
                'corpus': 'Represent the Webmaster answer: ',
            },
            'CQADupstackEnglishRetrieval':  {
                'query': 'Represent the English question for retrieving documents: ',
                'corpus': 'Represent the English answer for retrieval: ',
            },
            'CQADupstackGamingRetrieval': {
                'query': 'Represent the Gaming question for retrieving answers: ',
                'corpus': 'Represent the Gaming answer for retrieval: ',
            },
            'CQADupstackGisRetrieval': {
                'query': 'Represent the Gis question for retrieving answers: ',
                'corpus': 'Represent the Gis answer for retrieval: ',
            },
            'CQADupstackUnixRetrieval': {
                'query': 'Represent the Unix question for retrieving answers: ',
                'corpus': 'Represent the Unix answer for retrieval: ',
            },
            'CQADupstackMathematicaRetrieval': {
                'query': 'Represent the Mathematical question for retrieving answers: ',
                'corpus': 'Represent the Mathematical answer for retrieval: ',
            },
            'CQADupstackStatsRetrieval': {
                'query': 'Represent the Statistical question for retrieving answers: ',
                'corpus': 'Represent the Statistical answer for retrieval: ',
            },
            'CQADupstackPhysicsRetrieval': {
                'query': 'Represent the Physics question for retrieving answers: ',
                'corpus': 'Represent the Physics answer for retrieval: ',
            },
            'CQADupstackProgrammersRetrieval': {
                'query': 'Represent the Programming question for retrieving answers: ',
                'corpus': 'Represent the Programming answer for retrieval: ',
            },
            'CQADupstackAndroidRetrieval': {
                'query': 'Represent the Android question for retrieving answers: ',
                'corpus': 'Represent the Android answer for retrieval: ',
            },
            'CQADupstackWordpressRetrieval': {
                'query': 'Represent the Wordpress question for retrieving answers: ',
                'corpus': 'Represent the Wordpress answer for retrieval: ',
            },
            'FiQA2018': {
                'query': 'Represent the finance question for retrieving the supporting answers: ',
                'corpus': 'Represent the finance answer for retrieval: ',
            },
        },
    },
    'bge-large-en-v1.5': {
        'Retrieval': {
            'SciFact': {
                'query': 'Represent this sentence for searching relevant passages: ',
                'corpus': '',
            },
            'FiQA2018': {
                'query': 'Represent this sentence for searching relevant passages: ',
                'corpus': '',
            },
            'NFCorpus': {
                'query': 'Represent this sentence for searching relevant passages: ',
                'corpus': '',
            },
            'SCIDOCS': {
                'query': 'Represent this sentence for searching relevant passages: ',
                'corpus': '',
            },
            'TRECCOVID': {
                'query': 'Represent this sentence for searching relevant passages: ',
                'corpus': '',
            },
            'Touche2020': {
                'query': 'Represent this sentence for searching relevant passages: ',
                'corpus': '',
            },
            'DBPedia': {
                'query': 'Represent this sentence for searching relevant passages: ',
                'corpus': '',
            },
        }
    },
    'e5-mistral-7b-instruct': {
        "Classification":{
            "AmazonCounterfactualClassification":"Instruct: Classify a given Amazon customer review text as either counterfactual or not-counterfactual\nQuery: ",
            "AmazonPolarityClassification":"Instruct: Classify Amazon reviews into positive or negative sentiment\nQuery: ",
            "AmazonReviewsClassification":"Instruct: Classify the given Amazon review into its appropriate rating category\nQuery: ",
            "Banking77Classification":"Instruct: Given a online banking query, find the corresponding intents\nQuery: ",
            "EmotionClassification":"Instruct: Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise\nQuery: ",
            "ImdbClassification":"Instruct: Classify the sentiment expressed in the given movie review text from the IMDB dataset\nQuery: ",
            "MassiveIntentClassification":"Instruct: Given a user utterance as query, find the user intents\nQuery: ",
            "MassiveScenarioClassification":"Instruct: Given a user utterance as query, find the user scenarios\nQuery: ",
            "MTOPDomainClassification":"Instruct: Classify the intent domain of the given utterance in task-oriented conversation\nQuery: ",
            "MTOPIntentClassification":"Instruct: Classify the intent of the given utterance in task-oriented conversation\nQuery: ",
            "ToxicConversationsClassification":"Instruct: Classify the given comments as either toxic or not toxic\nQuery: ",
            "TweetSentimentExtractionClassification":"Instruct: Classify the sentiment of a given tweet as either positive, negative, or neutral\nQuery: "
        },
        "Clustering":{
            "ArxivClusteringP2P":"Instruct: Identify the main and secondary category of Arxiv papers based on the titles and abstracts\nQuery: ",
            "ArxivClusteringS2S":"Instruct: Identify the main and secondary category of Arxiv papers based on the titles\nQuery: ",
            "BiorxivClusteringP2P":"Instruct: Identify the main category of Biorxiv papers based on the titles and abstracts\nQuery: ",
            "BiorxivClusteringS2S":"Instruct: Identify the main category of Biorxiv papers based on the titles\nQuery: ",
            "MedrxivClusteringP2P":"Instruct: Identify the main category of Medrxiv papers based on the titles and abstracts\nQuery: ",
            "MedrxivClusteringS2S":"Instruct: Identify the main category of Medrxiv papers based on the titles\nQuery: ",
            "RedditClustering":"Instruct: Identify the topic or theme of Reddit posts based on the titles\nQuery: ",
            "RedditClusteringP2P":"Instruct: Identify the topic or theme of Reddit posts based on the titles and posts\nQuery: ",
            "StackExchangeClustering":"Instruct: Identify the topic or theme of StackExchange posts based on the titles\nQuery: ",
            "StackExchangeClusteringP2P":"Instruct: Identify the topic or theme of StackExchange posts based on the given paragraphs\nQuery: ",
            "TwentyNewsgroupsClustering":"Instruct: Identify the topic or theme of the given news articles\nQuery: "
        },
        "PairClassification":{
            "SprintDuplicateQuestions":"Instruct: Retrieve duplicate questions from Sprint forum\nQuery: ",
            "TwitterSemEval2015":"Instruct: Retrieve tweets that are semantically similar to the given tweet\nQuery: ",
            "TwitterURLCorpus":"Instruct: Retrieve tweets that are semantically similar to the given tweet\nQuery: "
        },
        "Reranking":{
            "AskUbuntuDupQuestions":"Instruct: Retrieve duplicate questions from AskUbuntu forum\nQuery: ",
            "MindSmallReranking":"Instruct: Retrieve relevant news articles based on user browsing history\nQuery: ",
            "SciDocsRR":"Instruct: Given a title of a scientific paper, retrieve the titles of other relevant papers\nQuery: ",
            "StackOverflowDupQuestions":"Instruct: Retrieve duplicate questions from StackOverflow forum\nQuery: "
        },
        "Retrieval":{
            "ArguAna":{
                "query":"Instruct: Given a claim, find documents that refute the claim\nQuery: ",
                "corpus":""
            },
            "ClimateFEVER":{
                "query":"Instruct: Given a claim about climate change, retrieve documents that support or refute the claim\nQuery: ",
                "corpus":""
            },
            "CQADupstackAndroidRetrieval":{
                "query":"Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
                "corpus":""
            },
            "CQADupstackEnglishRetrieval":{
                "query":"Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
                "corpus":""
            },
            "CQADupstackGamingRetrieval":{
                "query":"Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
                "corpus":""
            },
            "CQADupstackGisRetrieval":{
                "query":"Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
                "corpus":""
            },
            "CQADupstackMathematicaRetrieval":{
                "query":"Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
                "corpus":""
            },
            "CQADupstackPhysicsRetrieval":{
                "query":"Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
                "corpus":""
            },
            "CQADupstackProgrammersRetrieval":{
                "query":"Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
                "corpus":""
            },
            "CQADupstackStatsRetrieval":{
                "query":"Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
                "corpus":""
            },
            "CQADupstackTexRetrieval":{
                "query":"Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
                "corpus":""
            },
            "CQADupstackUnixRetrieval":{
                "query":"Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
                "corpus":""
            },
            "CQADupstackWebmastersRetrieval":{
                "query":"Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
                "corpus":""
            },
            "CQADupstackWordpressRetrieval":{
                "query":"Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
                "corpus":""
            },
            "DBPedia":{
                "query":"Instruct: Given a query, retrieve relevant entity descriptions from DBPedia\nQuery: ",
                "corpus":""
            },
            "FEVER":{
                "query":"Instruct: Given a claim, retrieve documents that support or refute the claim\nQuery: ",
                "corpus":""
            },
            "FiQA2018":{
                "query":"Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: ",
                "corpus":""
            },
            "HotpotQA":{
                "query":"Instruct: Given a multi-hop question, retrieve documents that can help answer the question\nQuery: ",
                "corpus":""
            },
            "MSMARCO":{
                "query":"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
                "corpus":""
            },
            "NFCorpus":{
                "query":"Instruct: Given a question, retrieve relevant documents that best answer the question\nQuery: ",
                "corpus":""
            },
            "NQ":{
                "query":"Instruct: Given a question, retrieve Wikipedia passages that answer the question\nQuery: ",
                "corpus":""
            },
            "QuoraRetrieval":{
                "query":"Instruct: Given a question, retrieve questions that are semantically equivalent to the given question\nQuery: ",
                "corpus":""
            },
            "SCIDOCS":{
                "query":"Instruct: Given a scientific paper title, retrieve paper abstracts that are cited by the given paper\nQuery: ",
                "corpus":""
            },
            "SciFact":{
                "query":"Instruct: Given a scientific claim, retrieve documents that support or refute the claim\nQuery: ",
                "corpus":""
            },
            "Touche2020":{
                "query":"Instruct: Given a question, retrieve detailed and persuasive arguments that answer the question\nQuery: ",
                "corpus":""
            },
            "TRECCOVID":{
                "query":"Instruct: Given a query on COVID-19, retrieve documents that answer the query\nQuery: ",
                "corpus":""
            }
        },
        "STS":{
            "BIOSSES":"Instruct: Retrieve semantically similar text.\nQuery: ",
            "SICK-R":"Instruct: Retrieve semantically similar text.\nQuery: ",
            "STS12":"Instruct: Retrieve semantically similar text.\nQuery: ",
            "STS13":"Instruct: Retrieve semantically similar text.\nQuery: ",
            "STS14":"Instruct: Retrieve semantically similar text.\nQuery: ",
            "STS15":"Instruct: Retrieve semantically similar text.\nQuery: ",
            "STS16":"Instruct: Retrieve semantically similar text.\nQuery: ",
            "STS17":"Instruct: Retrieve semantically similar text.\nQuery: ",
            "STS22":"Instruct: Retrieve semantically similar text.\nQuery: ",
            "STSBenchmark":"Instruct: Retrieve semantically similar text.\nQuery: "
        },
        "Summarization":{
            "SummEval":"Instruct: Given a news summary, retrieve other semantically similar summaries\nQuery: "
        }
    }
}

SET_TO_TASK_TO_DS_TO_SHOTS = {
    "e5": {
        "Classification": {
            "Banking77Classification": [
                "I am still waiting on my card?", 
                "card_arrival"
            ],
            "EmotionClassification": [
                "ive been feeling a little burdened lately wasnt sure why that was", 
                "sadness"
            ],
            "ImdbClassification": [
                "If only to avoid making this type of film in the future. This film is interesting as an experiment but tells no cogent story.<br /><br />One might feel virtuous for sitting thru it because it touches on so many IMPORTANT issues but it does so without any discernable motive. The viewer comes away with no new perspectives (unless one comes up with one while one's mind wanders, as it will invariably do during this pointless film).<br /><br />One might better spend one's time staring out a window at a tree growing.<br /><br />", 
                "negative"
            ],
        },
        "Clustering": {
            "BiorxivClusteringS2S": [
                "Association of CDH11 with ASD revealed by matched-gene co-expression analysis and mouse behavioral studies",
                "neuroscience",
            ],
        },
        "PairClassification": {
            "SprintDuplicateQuestions": [
                "Why is it impossible for me to find a easy way to send a picture with text on my Kyocera DuraCore ?",
                "Send or receive a picture with text - Kyocera DuraCore",
            ],
            "TwitterSemEval2015": [
                "The Ending to 8 Mile is my fav part of the whole movie",
                "Those last 3 battles in 8 Mile are THE shit",
            ],
            "TwitterURLCorpus": [
                "Liberals , dont let Donald Trump tarnish L.L. Beans sterling brand reputation ",
                "Liberals, Don&rsquo;t Let Donald Trump Tarnish L.L. Bean&rsquo;s Sterling Brand Reputation",
            ],            
        },
        "Reranking": {
            "AskUbuntuDupQuestions": {
                "query": [
                    "what is a short cut i can use to switch applications ?",
                    "keyboard short cut for switching between two or more instances of the same application ?",
                ],
                "corpus": [
                    "keyboard short cut for switching between two or more instances of the same application ?",
                    "what is a short cut i can use to switch applications ?",
                ],
            },
        },
        "Retrieval": {
            'ArguAna': {
                'query': [
                    "People will die if we don’t do animal testing Every year, 23 new drugs are introduced in the UK alone.[13] Almost all will be tested on animals. A new drug will be used for a long time. Think of all the people saved by the use of penicillin. If drugs cost more to test, that means drug companies will develop less. This means more people suffering and dying",
                    "animals science science general ban animal testing junior" + " " +  "Many of these drugs are “me too” drugs – ones with a slight change that doesn’t make much difference to an existing drug. [14] So often the benefits from animal testing are marginal, and even if there was a slight increase in human suffering, it would be worth it based on the animal suffering saved.",
                ],
                'corpus': [
                    "animals science science general ban animal testing junior" + " " +  "Many of these drugs are “me too” drugs – ones with a slight change that doesn’t make much difference to an existing drug. [14] So often the benefits from animal testing are marginal, and even if there was a slight increase in human suffering, it would be worth it based on the animal suffering saved.",
                    "People will die if we don’t do animal testing Every year, 23 new drugs are introduced in the UK alone.[13] Almost all will be tested on animals. A new drug will be used for a long time. Think of all the people saved by the use of penicillin. If drugs cost more to test, that means drug companies will develop less. This means more people suffering and dying",
                ],
            },
            'SCIDOCS': {
                'query': [
                    "A Direct Search Method to solve Economic Dispatch Problem with Valve-Point Effect",
                    "A Hybrid EP and SQP for Dynamic Economic Dispatch with Nonsmooth Fuel Cost Function" + " " + "Dynamic economic dispatch (DED) is one of the main functions of power generation operation and control. It determines the optimal settings of generator units with predicted load demand over a certain period of time. The objective is to operate an electric power system most economically while the system is operating within its security limits. This paper proposes a new hybrid methodology for solving DED. The proposed method is developed in such a way that a simple evolutionary programming (EP) is applied as a based level search, which can give a good direction to the optimal global region, and a local search sequential quadratic programming (SQP) is used as a fine tuning to determine the optimal solution at the final. Ten units test system with nonsmooth fuel cost function is used to illustrate the effectiveness of the proposed method compared with those obtained from EP and SQP alone.",
                ],
                'corpus': [
                    "A Hybrid EP and SQP for Dynamic Economic Dispatch with Nonsmooth Fuel Cost Function" + " " + "Dynamic economic dispatch (DED) is one of the main functions of power generation operation and control. It determines the optimal settings of generator units with predicted load demand over a certain period of time. The objective is to operate an electric power system most economically while the system is operating within its security limits. This paper proposes a new hybrid methodology for solving DED. The proposed method is developed in such a way that a simple evolutionary programming (EP) is applied as a based level search, which can give a good direction to the optimal global region, and a local search sequential quadratic programming (SQP) is used as a fine tuning to determine the optimal solution at the final. Ten units test system with nonsmooth fuel cost function is used to illustrate the effectiveness of the proposed method compared with those obtained from EP and SQP alone.",
                    "A Direct Search Method to solve Economic Dispatch Problem with Valve-Point Effect",
                ],
            },            
        },
        "STS": {
            'STS12': [	
                "Counties with population declines will be Vermillion, Posey and Madison.",
                "Vermillion, Posey and Madison County populations will decline.",
            ],
        },
        "Summarization": {
            "SummEval": [
                "Mexican restaurant has decided to tap into $70 billion food delivery market. Fast-casual chain will work with the Postmates app to allow mobile orders. App works in similar way to Uber, using hired drivers to deliver the food. But the chain will add a 9% service charge - on top of Postmates' $5 rate.",
                "chipotle has decided to tap into the $ 70 billion food delivery market by teaming up with an app to bring burritos straight to customers ' doors . the fast-casual chain will work with the postmates app to begin offering delivery for online and mobile orders in 67 cities . the restaurant plans to add a nine per cent service charge - with the delivery fees for postmates beginning at $ 5 and up depending on distance and demand .",
            ],
        },
    },
    "medi2": {
        "Classification": {
            "Banking77Classification": [
                "I am still waiting on my card?",
                "What can I do if my card still hasn't arrived after 2 weeks?",
            ],
            "EmotionClassification": [
                "ive been feeling a little burdened lately wasnt sure why that was",
                "i feel like i have to make the suffering i m seeing mean something",
            ],
            "ImdbClassification": [
                "If only to avoid making this type of film in the future. This film is interesting as an experiment but tells no cogent story.<br /><br />One might feel virtuous for sitting thru it because it touches on so many IMPORTANT issues but it does so without any discernable motive. The viewer comes away with no new perspectives (unless one comes up with one while one's mind wanders, as it will invariably do during this pointless film).<br /><br />One might better spend one's time staring out a window at a tree growing.<br /><br />",
                "The silent one-panel cartoon Henry comes to Fleischer Studios, billed as \"The world's funniest human\" in this dull little cartoon. Betty, long past her prime, thanks to the Production Code, is running a pet shop and leaves Henry in charge for far too long -- five minutes. A bore.",
            ],
        },
        "Clustering": {
            "BiorxivClusteringS2S": [
                "Association of CDH11 with ASD revealed by matched-gene co-expression analysis and mouse behavioral studies", 
                "Gliotransmission of D-serine promotes thirst-directed behaviors in Drosophila",
            ],
            "TwentyNewsgroupsClustering": [
                "Need to find out number to a phone line", 
                "what to do with old 256k SIMMs?",
            ],
        },
        "PairClassification": {
            "SprintDuplicateQuestions": [
                "Why is it impossible for me to find a easy way to send a picture with text on my Kyocera DuraCore ?",
                "Send or receive a picture with text - Kyocera DuraCore",
            ],
            "TwitterSemEval2015": [
                "The Ending to 8 Mile is my fav part of the whole movie",
                "Those last 3 battles in 8 Mile are THE shit",
            ],
            "TwitterURLCorpus": [
                "Liberals , dont let Donald Trump tarnish L.L. Beans sterling brand reputation ",
                "Liberals, Don&rsquo;t Let Donald Trump Tarnish L.L. Bean&rsquo;s Sterling Brand Reputation",
            ],
        },
        "Reranking": {
            "AskUbuntuDupQuestions": {
                "query": [
                    "what is a short cut i can use to switch applications ?",
                    "keyboard short cut for switching between two or more instances of the same application ?",
                ],
                "corpus": [
                    "keyboard short cut for switching between two or more instances of the same application ?",
                    "what is a short cut i can use to switch applications ?",
                ],
            },
        },
        "Retrieval": {
            'ArguAna': {
                'query': [
                    "People will die if we don’t do animal testing Every year, 23 new drugs are introduced in the UK alone.[13] Almost all will be tested on animals. A new drug will be used for a long time. Think of all the people saved by the use of penicillin. If drugs cost more to test, that means drug companies will develop less. This means more people suffering and dying",
                    "animals science science general ban animal testing junior" + " " +  "Many of these drugs are “me too” drugs – ones with a slight change that doesn’t make much difference to an existing drug. [14] So often the benefits from animal testing are marginal, and even if there was a slight increase in human suffering, it would be worth it based on the animal suffering saved.",
                ],
                'corpus': [
                    "animals science science general ban animal testing junior" + " " +  "Many of these drugs are “me too” drugs – ones with a slight change that doesn’t make much difference to an existing drug. [14] So often the benefits from animal testing are marginal, and even if there was a slight increase in human suffering, it would be worth it based on the animal suffering saved.",
                    "People will die if we don’t do animal testing Every year, 23 new drugs are introduced in the UK alone.[13] Almost all will be tested on animals. A new drug will be used for a long time. Think of all the people saved by the use of penicillin. If drugs cost more to test, that means drug companies will develop less. This means more people suffering and dying",
                ],
            },
            'SCIDOCS': {
                'query': [
                    "A Direct Search Method to solve Economic Dispatch Problem with Valve-Point Effect",
                    "A Hybrid EP and SQP for Dynamic Economic Dispatch with Nonsmooth Fuel Cost Function" + " " + "Dynamic economic dispatch (DED) is one of the main functions of power generation operation and control. It determines the optimal settings of generator units with predicted load demand over a certain period of time. The objective is to operate an electric power system most economically while the system is operating within its security limits. This paper proposes a new hybrid methodology for solving DED. The proposed method is developed in such a way that a simple evolutionary programming (EP) is applied as a based level search, which can give a good direction to the optimal global region, and a local search sequential quadratic programming (SQP) is used as a fine tuning to determine the optimal solution at the final. Ten units test system with nonsmooth fuel cost function is used to illustrate the effectiveness of the proposed method compared with those obtained from EP and SQP alone.",
                ],
                'corpus': [
                    "A Hybrid EP and SQP for Dynamic Economic Dispatch with Nonsmooth Fuel Cost Function" + " " + "Dynamic economic dispatch (DED) is one of the main functions of power generation operation and control. It determines the optimal settings of generator units with predicted load demand over a certain period of time. The objective is to operate an electric power system most economically while the system is operating within its security limits. This paper proposes a new hybrid methodology for solving DED. The proposed method is developed in such a way that a simple evolutionary programming (EP) is applied as a based level search, which can give a good direction to the optimal global region, and a local search sequential quadratic programming (SQP) is used as a fine tuning to determine the optimal solution at the final. Ten units test system with nonsmooth fuel cost function is used to illustrate the effectiveness of the proposed method compared with those obtained from EP and SQP alone.",
                    "A Direct Search Method to solve Economic Dispatch Problem with Valve-Point Effect",
                ],
            },
        },
        "STS": {
            'STS12': [	
                "Counties with population declines will be Vermillion, Posey and Madison.",
                "Vermillion, Posey and Madison County populations will decline.",
            ],
        },
        "Summarization": {
            'SummEval': {
                "query": [
                    "Mexican restaurant has decided to tap into $70 billion food delivery market. Fast-casual chain will work with the Postmates app to allow mobile orders. App works in similar way to Uber, using hired drivers to deliver the food. But the chain will add a 9% service charge - on top of Postmates' $5 rate.",
                    "chipotle has decided to tap into the $ 70 billion food delivery market by teaming up with an app to bring burritos straight to customers ' doors . the fast-casual chain will work with the postmates app to begin offering delivery for online and mobile orders in 67 cities . the restaurant plans to add a nine per cent service charge - with the delivery fees for postmates beginning at $ 5 and up depending on distance and demand .",
                ],
                "corpus": [
                    "chipotle has decided to tap into the $ 70 billion food delivery market by teaming up with an app to bring burritos straight to customers ' doors . the fast-casual chain will work with the postmates app to begin offering delivery for online and mobile orders in 67 cities . the restaurant plans to add a nine per cent service charge - with the delivery fees for postmates beginning at $ 5 and up depending on distance and demand .",
                    "Mexican restaurant has decided to tap into $70 billion food delivery market. Fast-casual chain will work with the Postmates app to allow mobile orders. App works in similar way to Uber, using hired drivers to deliver the food. But the chain will add a 9% service charge - on top of Postmates' $5 rate.",
                ],
            },
        },
    },
}

QUICK_EVAL = [
    # Classification
    "Banking77Classification",
    "EmotionClassification",
    # Clustering
    "MedrxivClusteringS2S",
    # PairClassification
    "TwitterSemEval2015",
    # Reranking
    "AskUbuntuDupQuestions",
    # Retrieval
    "ArguAna",
    "NFCorpus",
    "SciFact",
    # STS
    "BIOSSES",
    "STS17",
    "STSBenchmark",
    # Summarization
    "SummEval",
]

DTYPE_TO_TORCH_DTYPE = {
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
    'float16': torch.float16,
}

def get_gpus_max_memory(max_memory):
    max_memory = {i: max_memory for i in range(torch.cuda.device_count())}
    return max_memory

def gritlm_instruction_format(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

def zephyr_instruction_format(instruction):
    return "<|user|>\n" + instruction + "</s>\n<|assistant|>\n"

def tulu_instruction_format(instruction):
    return "<|user|>\n" + instruction + "\n<|assistant|>\n"

def mistral_instruction_format(instruction):
    return "[INST] " + instruction + " [/INST] "

NAME_TO_FUNC = {
    "gritlm": gritlm_instruction_format,
    "zephyr": zephyr_instruction_format,
    "tulu": tulu_instruction_format,
    "mistral": mistral_instruction_format,
}

SET_TO_FEWSHOT_PROMPT = {
    "e5": {
        "Retrieval": '\n\nFor example given "{}", you should retrieve "{}"',
        "Other": '\n\nFor example given "{}", it would match with "{}"',
    },
    "medi2": {
        "Retrieval": '\n\nThe provided query could be "{}" and the positive "{}"',
        "Other": '\n\nThe provided query could be "{}" and the positive "{}"',
    },
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="GritLM/GritLM-7B", type=str)
    parser.add_argument('--attn_implementation', default='sdpa', type=str, help="eager/sdpa/flash_attention_2")
    parser.add_argument('--attn', default='bbcc', type=str, help="only first two letters matter for embedding")
    parser.add_argument('--task_types', default=None, help="Comma separated. Default is None i.e. running all tasks")
    parser.add_argument('--task_names', default=None, help="Comma separated. Default is None i.e. running all tasks")
    parser.add_argument('--instruction_set', default="e5", type=str, help="Instructions to use")
    parser.add_argument('--instruction_format', default="gritlm", type=str, help="Formatting to use")
    parser.add_argument('--no_instruction', action='store_true', help="Do not use instructions")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_length', default=None, type=int)
    parser.add_argument('--num_shots', default=None, type=int)
    parser.add_argument('--dtype', default='bfloat16', type=str)
    parser.add_argument('--output_folder', default=None, type=str)
    parser.add_argument('--overwrite_results', action='store_true')
    parser.add_argument('--pipeline_parallel', action='store_true')
    parser.add_argument('--embedding_head', default=None, type=str)
    parser.add_argument('--pooling_method', default='mean', type=str)
    parser.add_argument('--save_qrels', action='store_true')
    parser.add_argument('--top_k', default=10, type=int)    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    # Quick skip if exists
    model_name = args.model_name_or_path.rstrip('/').split('/')[-1]
    output_folder = args.output_folder if args.output_folder else f"results/{model_name}"
    if (args.task_names is not None) and (len(args.task_names.split(",")) == 1) and os.path.exists(f"{output_folder}/{args.task_names.split(',')[0]}.json"):
        print(f"Skipping {args.task_names.split(',')[0]}")
        exit()

    kwargs = {
        "model_name_or_path": args.model_name_or_path,
        # Normalizing embeddings will harm the performance of classification task
        # as it does not use the cosine similarity
        # For other tasks, cosine similarity is used in the evaluation, 
        # so embeddings are automatically normalized
        "normalized": False,
        "torch_dtype": DTYPE_TO_TORCH_DTYPE.get(args.dtype, torch.bfloat16),
        "mode": "embedding",
        "pooling_method": args.pooling_method,
        "attn_implementation": args.attn_implementation,
        "attn": args.attn,
    }

    if args.pipeline_parallel:
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = get_gpus_max_memory("50GB")
        kwargs["offload_folder"] = "offload"

    if any([x in args.model_name_or_path for x in ["instructor"]]):
        assert kwargs["pooling_method"] == "mean"
    elif any([x in args.model_name_or_path for x in ["bge"]]):
        assert kwargs["pooling_method"] == "cls"
    
    if args.pooling_method == "lasttoken":
        kwargs["embed_eos"] = "</e>"
    if args.embedding_head:
        kwargs["projection"] = args.embedding_head

    model = GritLM(**kwargs)
    if args.embedding_head:
        model.load_state_dict(
            torch.load(args.model_name_or_path + "/embedding_head.bin"), strict=False,
        )
        model.projection.to(model.device)

    if os.getenv("BIDIRECTIONAL_ATTN", False):
        model.model.padding_idx = model.tokenizer.pad_token_id
        if hasattr(model.model, "model"):
            model.model.model.padding_idx = model.tokenizer.pad_token_id
        if hasattr(model.model, "module"):
            model.model.module.padding_idx = model.tokenizer.pad_token_id            

    kwargs = {"task_langs": ['en']}
    if args.task_names:
        kwargs["tasks"] = args.task_names.split(",")
    elif args.task_types:
        kwargs["task_types"] = args.task_types.split(",")
    tasks = [(t.metadata.name, t.metadata.type) for t in MTEB(**kwargs).tasks]
    
    if args.max_length is not None:
        model.encode = partial(model.encode, max_length=args.max_length)

    for (task_name, task_type) in tasks:
        if task_name in ['MSMARCOv2', 'BigPatentClustering']:
            print('Skipping task: ' + task_name)
            continue
        if not args.no_instruction:
            if task_name.startswith("CQADupstack") and \
                "CQADupstackRetrieval" in SET_TO_TASK_TO_DS_TO_PROMPT[args.instruction_set][task_type]:
                instruction = SET_TO_TASK_TO_DS_TO_PROMPT[args.instruction_set][task_type]["CQADupstackRetrieval"]
            else:
                if task_name not in SET_TO_TASK_TO_DS_TO_PROMPT[args.instruction_set][task_type]:
                    print('Skipping task: ' + task_name)
                    continue
                instruction = SET_TO_TASK_TO_DS_TO_PROMPT[args.instruction_set][task_type][task_name]
            if isinstance(instruction, dict):
                if args.num_shots is not None:
                    instruction = {
                        k: v + SET_TO_FEWSHOT_PROMPT[args.instruction_set]["Retrieval"].format(
                            *SET_TO_TASK_TO_DS_TO_SHOTS[args.instruction_set][task_type][task_name][k]
                        ) if v else v for k, v in instruction.items()
                    }
                instruction = {k: NAME_TO_FUNC[args.instruction_format](v.strip(": \n")) for k, v in instruction.items()}
            else:
                if args.num_shots is not None:
                    instruction = instruction + SET_TO_FEWSHOT_PROMPT[args.instruction_set]["Other"].format(
                        *SET_TO_TASK_TO_DS_TO_SHOTS[args.instruction_set][task_type][task_name]
                    )
                instruction = NAME_TO_FUNC[args.instruction_format](instruction.strip(": \n"))
            print(f"{model_name} instruction for {task_name}: ", instruction)
            if isinstance(instruction, dict):
                model.encode_queries = partial(model.encode_queries, instruction=instruction['query'])
                model.encode_corpus = partial(model.encode_corpus, instruction=instruction['corpus'])
            else:
                model.encode = partial(model.encode, instruction=instruction)
        eval_splits = ["test" if task_name not in ['MSMARCO'] else 'dev']
        evaluation = MTEB(tasks=[task_name], task_langs=['en'])
        evaluation.run(
            model,
            output_folder=output_folder,
            eval_splits=eval_splits,
            batch_size=args.batch_size,
            save_qrels=args.save_qrels,
            top_k=args.top_k,
            overwrite_results=args.overwrite_results,
        )
