import json
import glob
import os
import time
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

class OKVQADataset():
    """
    A OKVQA data example in json file:
    {
        "img_id": "2375429",
        "label": {
            "pipe": 1.0
        },
        "question_id": "07333408",
        "sent": "What is on the white wall?"
    }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open("data/okvqa/kn_%s.json" % split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open("data/okvqa/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/okvqa/trainval_label2ans.json"))



        assert len(self.ans2label) == len(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)

class OKVQABufferLoader():
    def __init__(self):
        self.key2data = {}

    def _load_npy_files(self, path, topk=None):
        data = []
        start_time = time.time()
        fnames = []
        print("Start to load Faster-RCNN detected objects from %s" % path)
        # fnames = glob.glob(os.path.join(path, "*"))
        with open(path, 'r') as f:
            fnames = json.load(f)
        for fname in fnames:
            imfeat = dict()
            img_info = np.load(fname, allow_pickle=True)
            img_info = img_info.tolist()
            imfeat['img_h'] = int(img_info['image_height'])
            imfeat['img_w'] = int(img_info['image_width'])
            imfeat['num_boxes'] = img_info['num_boxes']
            imfeat['boxes'] = img_info['bbox']#[1:]
            imfeat['objects_id'] = img_info['objects']
            imfeat['objects_conf'] = img_info['scores']
            imfeat['features'] = img_info['features']
            imfeat['img_id'] = img_info['image_id']
            # print(imfeat['img_id'])
            
            data.append(imfeat)
            if topk is not None and len(data) == topk:
                break

            elapsed_time = time.time() - start_time
        print("Loaded %d images in path %s in %d seconds." % (len(data), path, elapsed_time))
        return data

    def load_data(self, name, number):
        if name == 'testdev':
            path = "data/okvqa/testdev_img_list.json"
        else:
            path = "data/okvqa/train_img_list.json"
        key = "%s_%d" % (path, number)
        if key not in self.key2data:
            self.key2data[key] = self._load_npy_files(
                path,
                topk=number
            )
        return self.key2data[key]


okvqa_buffer_loader = OKVQABufferLoader()

"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
class OKVQATorchDataset(Dataset):
    def __init__(self, dataset: OKVQADataset):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = -1

        # Loading detection features to img_data
        # Since images in train and valid both come from Visual Genome,
        # buffer the image loading to save memory.
        img_data = []
        if 'testdev' in dataset.splits or 'testdev_all' in dataset.splits:     # Always loading all the data in testdev
            img_data.extend(okvqa_buffer_loader.load_data('testdev', -1))
        else:
            img_data.extend(okvqa_buffer_loader.load_data('train', topk))
        self.imgid2img = {}
        for img_datum in img_data:
            # print('img_in_feat:', img_datum['img_id'])
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            # print('img_in_data:', datum['img_id'])
            if datum['img_id'] in self.imgid2img:
                # print(datum['img_id'])
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        boxes = img_info['boxes'].copy()
        feats = img_info['features'].copy()
        assert len(boxes) == len(feats) == obj_num

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        #print(img_id)
        #print(img_w, img_h)
        #print(boxes)
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        for bd_box in boxes:
            if bd_box[0] < -0.0001:
                bd_box[0] = 0
            if bd_box[0] > 1:
                bd_box[0] = 0.99
            if bd_box[2] > 1:
                bd_box[2] = 1
            if bd_box[1] < -0.0001:
                bd_box[1] = 0
            if bd_box[1] > 1:
                bd_box[1] = 0.99
            if bd_box[3] > 1:
                bd_box[3] = 1
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # Create target
        if 'label' in datum:
            label = datum['label']
            '''
            if 'train' in self.raw_dataset.splits:
                if not datum['has_answer']:
                    label = 'unkn'
            '''
            target = torch.zeros(self.raw_dataset.num_answers)
            '''
            for ans, score in label.items():
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
            '''
            for ans in label:
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = 1
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques


class OKVQAEvaluator:
    def __init__(self, dataset: OKVQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        num_ans_list = [100, 80, 50, 20, 15, 10, 8, 5, 3, 1]
        if type(list(quesid2ans.values())[0]) == dict:
            score = {x: 0 for x in num_ans_list}
            freq = {x: 0 for x in num_ans_list}
            top5 = {x: 0 for x in num_ans_list}
        else:
            score = 0
            freq = 0
            top5 = 0
        for quesid, anss in quesid2ans.items():
            qid = int(quesid)
            datum = self.dataset.id2datum[qid]
            label = datum['label']
            anss = [{'score': anss['score'][x], 'label': anss['label'][x], 'sent': anss['sent'][x]} for x in range(len(anss['score']))]
            anss = [x for x in anss if 'unkn' != x['label']]
            if not anss:
                anss = [{'label': 'untkn', 'score': 0., 'sent': ''}]
            anss.sort(key=lambda x: x['score'], reverse=True)
            if type(anss) == list:
                for num_ans in num_ans_list:
                    anss = anss[:num_ans]
                    ans = anss[0]['label']
                    t_ans = [ans['label'] for ans in anss[:5]]
                    ans_label = [ans['label'] for ans in anss]
                    f_ans = max(set(ans_label), key = ans_label.count)
                    if ans in label:
                        score[num_ans] += label[ans]
                    if f_ans in label:
                        freq[num_ans] += label[f_ans]
                    for t_item in t_ans:
                        if t_item in label:
                            top5[num_ans] += label[t_item]
                            break
            else:
                ans = anss
                if ans in label:
                    score += label[ans]
        if type(list(quesid2ans.values())[0]) == dict:
            return {x: y/len(quesid2ans) for x, y in score.items()}, {x: y/len(quesid2ans) for x, y in freq.items()}, {x: y/len(quesid2ans) for x, y in top5.items()}
        else:
            return score / len(quesid2ans), freq / len(quesid2ans), top5 / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump the result to a GQA-challenge submittable json file.
        GQA json file submission requirement:
            results = [result]
            result = {
                "questionId": str,      # Note: it's a actually an int number but the server requires an str.
                "prediction": str
            }
        :param quesid2ans: A dict mapping question id to its predicted answer.
        :param path: The file path to save the json file.
        :return:
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'questionId': ques_id.item() if not type(ques_id) == int else ques_id,
                    'prediction': ans['label']
                })
            json.dump(result, f, indent=4, sort_keys=True)

