from utils import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieve_kn_file', type=str, required = True, default=None,
                        help="retrieved knowledge of each question")
    args = parser.parse_args()
    data = load(args.retrieve_kn_file) 
    
    precision = {
        1:0,
        5:0,
        10:0,
        20:0,
        50:0,
        80:0,
        100:0
    }

    for s in data:
        for p in precision:
            for c in s['ctxs'][:p]:
                text = c['text'].lower()
                for ans in s['answers']:
                    if ans in text:
                        precision[p]+=1
                        break
    for k, v in precision.items():
        print("precision for top {} is {:.2f}".format(k, 100*v/(k*len(data))))

    recall = {
        1:0,
        5:0,
        10:0,
        20:0,
        50:0,
        80:0,
        100:0
    }
    for r in recall:
        for s in data:
            found = False
            for ans in s['answers']:
                for c in s['ctxs'][:r]:
                    text = c['text'].lower()
                    if ans in text:
                        recall[r]+=1
                        found = True
                        break
                if found:
                    break
    for k,v in recall.items():
        print("recall for top {} text is {:.2f}".format(k, 100*v/len(data)))

