from datasets import load_dataset

data_keys = ['triviaqa-train-multikilt', 'wow-train-multikilt', 'multi_lexsum', 'WikiAnswers', 'reddit-title-body', 'altlex', 'medmcqa', 'SimpleWiki', 'amazon-qa', 'eli5_question_answer', 'gooaq_pairs', 'squad_pairs', 'searchQA_top5_snippets', 'coco_captions', 'ccnews_title_text', 'pubmedqa', 'npr', 'S2ORC_title_abstract', 'wikihow', 'trex-train-multikilt', 'nq-train-multikilt', 'gigaword', 'cnn_dailymail', 'agnews', 'yahoo_answers_title_answer', 'fever-train-multikilt', 'flickr30k_captions', 'sentence-compression', 'zeroshot-train-multikilt', 'PAQ_pairs', 'hotpotqa-train-multikilt', 'scitldr', 'xsum', 'amazon_review_2018']

for key in data_keys:
    d = load_dataset(f"multi-train/{key}_1107",token='YOUR_HF_KEY',
                                            cache_dir='/home2/huggingface/datasets/v1107')['train']
    print(d)
