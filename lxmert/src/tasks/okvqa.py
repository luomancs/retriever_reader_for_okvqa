import os
import collections

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.okvqa_model import OKVQAModel
from tasks.okvqa_data import OKVQADataset, OKVQATorchDataset, OKVQAEvaluator


DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = OKVQADataset(splits)
    tset = OKVQATorchDataset(dset)
    evaluator = OKVQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class OKVQA:
    def __init__(self):
        self.train_tuple = get_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            valid_bsize = 2048 if args.multiGPU else 16
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize, shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        self.model = OKVQAModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Losses and optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = [[{"score": 0}]]
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                if args.mce_loss:
                    max_value, target = target.max(1)
                    loss = self.mce_loss(logit, target) * logit.size(1)
                else:
                    loss = self.bce_loss(logit, target)
                    loss = loss * logit.size(1)

                loss.backward()
                # print('loss: ', loss)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.)# 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans)[0] * 100.)
            evaluator.dump_result(quesid2ans, os.path.join(args.output, 'train_predict.json'))

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score[0][0] > best_valid[0][0]:
                    best_valid = valid_score
                    self.save("BEST")
                
                

                log_str = "Epoch %d: Valid %0.2f\n" % (epoch, valid_score[0] * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid[0] * 100.)

                '''
                log_str += "Epoch %d: Valid %0.2f Freq %0.2f Top5 %0.2f\n" % (epoch, valid_score[0][50] * 100., valid_score[1][50] * 100., valid_score[2][50] * 100.) + \
                           "Epoch %d: Best %0.2f Freq %0.2f Top5 %0.2f\n" % (epoch, best_valid[0][50] * 100., best_valid[1][50] * 100., best_valid[2][50] * 100.)
                '''

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, label = logit.max(dim=1)
                # score, label = logit.topk(10, dim=1)
                for qid, l, sc, st in zip(ques_id, label.cpu().numpy(), score.cpu().numpy(), sent):
                    ans = dset.label2ans[l]
                    # ans = [dset.label2ans[ll] for ll in l]
                    qid = int(qid)
                    '''
                    # quesid2ans[qid] = ans
                    quesid2ans[qid] = {'label': ans, 'score': sc}
                    '''
                    if qid not in quesid2ans:
                        quesid2ans[qid] = {'label': [ans], 'score': [float(sc)], 'sent': [st]}
                    else:
                        quesid2ans[qid]['label'].append(ans)
                        quesid2ans[qid]['score'].append(float(sc))
                        quesid2ans[qid]['sent'].append(st)

        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        for key in list(state_dict.keys()):
            if '.module' in key:
                state_dict[key.replace('.module', '')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    # Build Class
    okvqa = OKVQA()

    # Load Model
    if args.load is not None:
        okvqa.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'submit' in args.test:
            okvqa.predict(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'submit_predict.json')
            )
        if 'testdev' in args.test:
            result = okvqa.evaluate(
                get_tuple('testdev', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'testdev_predict.json')
            )
            print(result)
    else:
        # print("Train Oracle: %0.2f" % (okvqa.oracle_score(okvqa.train_tuple) * 100))
        # iter_wrapper = (lambda x: tqdm(x, total=len(okvqa.valid_tuple.loader))) if args.tqdm else (lambda x: x)
        print('Splits in Train data:', okvqa.train_tuple.dataset.splits)
        if okvqa.valid_tuple is not None:
            # for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(okvqa.valid_tuple.loader)):
                # print(ques_id)
            print('Splits in Valid data:', okvqa.valid_tuple.dataset.splits)
            # print("Valid Oracle: %0.2f" % (okvqa.oracle_score(okvqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        okvqa.train(okvqa.train_tuple, okvqa.valid_tuple)

