from transformers import pipeline
from collections import Counter
import tqdm
import json
import random
import torch
import argparse
from utils import *

def get_most_frequent_answer(answers:list):
    answers_list = [a['answer'].lower() for a in answers]
    ans= Counter(answers_list).most_common()[0][0]
    return ans

def evaluate(args, qa_pipeline):
    data = load(args.retrieve_kn_file)
    all_q = len(data)
    topk=100
    predicted_answer = []
    correct_top1 = {
        1:0,
        3:0,
        5:0,
        8:0,
        10:0,
        15:0,
        20:0,
        50:0,
        60:0,
        70:0,
        80:0,
        90:0,
        100:0
    }

    correct_most_frequence = {
        1:0,
        3:0,
        5:0,
        8:0,
        10:0,
        15:0,
        20:0,
        50:0,
        60:0,
        70:0,
        80:0,
        90:0,
        100:0
    }
    
    unanswerable =0
    for d in tqdm.tqdm(data):
        question = d['question'].split("?")
        caption = question[1]
        question = [question[0]+"?"]

        context = [ "unanswerable, "+caption+ " "+ c['text'] for c in d['ctxs'][:topk]]
        question = question * len(context)
        answers = qa_pipeline(question=question, context=context)

        gold = list(d['answers'].keys())
        good_answers = []
        if isinstance(answers, dict):
            answers = [answers]
        for a in answers:
            if a['answer'] != 'unanswerable':
                good_answers.append(a)
        answers=good_answers      
        predicted_answer.append(answers)
        if len(answers) == 0:
            unanswerable+=1
            continue

        for size in correct_top1:

            common = get_most_frequent_answer(answers[:size]).lower().rstrip().strip() 
            ans = sorted(answers[:size], key=lambda x: x['score'], reverse=True)
            top1 = ans[0]['answer'].lower().rstrip().strip()
            if top1 in gold:
                correct_top1[size]+=d['answers'][top1.lower().rstrip().strip()]
            if common in gold:
                correct_most_frequence[size]+=d['answers'][common.lower().rstrip().strip()]
        d['top1_prediction'] = top1 
        d['common_prediction'] = common
            
    for k, v in correct_top1.items():
            print("accuracy of using highest score strategy in the top{} text is {:.2f}".format(k,100*v/(len(data))))
    for k, v in correct_most_frequence.items():
            print("accuracy of using  most frequent strategy in the top{} text is {:.2f}".format(k,100*v/(len(data))))
    if args.prediction_save_path:
        dump(data, args.prediction_save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieve_kn_file', type=str, required = True, default=None,
                        help="retrieved knowledge of each question")
    
    parser.add_argument('--model_path', type=str, required = True, default=None,
                        help="the path to the pretrained extractive reader")
    parser.add_argument('--prediction_save_path', type=str, required = True, default=None,
                        help="the path to the save the prediction")
    
    parser.add_argument('--cuda_id', type=int, default=0,
                        help="using cuda if available, otherwise cpu")
    args = parser.parse_args()
    
    if not torch.cuda.is_available:
        args.cuda_id = -1
    
    qa_pipeline = pipeline("question-answering", model=args.model_path, tokenizer=args.model_path, device=args.cuda_id) 
    
    evaluate(args, qa_pipeline)
    