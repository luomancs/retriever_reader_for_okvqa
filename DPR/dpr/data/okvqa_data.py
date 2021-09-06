import collections
import csv
import glob
import logging
import os
import random
from typing import Dict, List, Tuple

import hydra
import jsonlines
import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor as T

from dpr.utils.data_utils import read_data_from_json_files, Tensorizer

from dpr.data.biencoder_data import (
    BiEncoderPassage,
    normalize_passage,
    normalize_question
)

logger = logging.getLogger(__name__)
BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])

from dpr.data.biencoder_data import Dataset

PROJ_PATH = dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class VisBiEncoderSample(object):
    query: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]
    img_id: str

class JsonQADataset(Dataset):
    def __init__(
        self,
        file: str,
        selector: DictConfig = None,
        special_token: str = None,
        encoder_type: str = None,
        shuffle_positives: bool = False,
        normalize: bool = False,
        query_special_suffix: str = None,
    ):
        super().__init__(
            selector,
            special_token=special_token,
            encoder_type=encoder_type,
            shuffle_positives=shuffle_positives,
            query_special_suffix=query_special_suffix,
        )
        self.file = file
        self.data_files = []
        self.data = []
        self.normalize = normalize
        logger.info("Data files: %s", self.data_files)

    def load_data(self):
        self.data_files = self.file
        
        if not os.path.exists(self.data_files):
            self.data_files = os.path.join(PROJ_PATH,self.data_files )
        print("read file", self.data_files )
        data = read_data_from_json_files(self.data_files)
        # filter those without positive ctx
        self.data = [r for r in data if len(r["ctxs"]) > 0]
        logger.info("Total cleaned data size: {}".format(len(self.data)))

    def __getitem__(self, index) :
        json_sample = self.data[index]
        r = VisBiEncoderSample()
        r.query = self._process_query(json_sample["question"])

        positive_ctxs = json_sample["ctxs"]
        negative_ctxs = (
            json_sample["negative_ctxs"] if "negative_ctxs" in json_sample else []
        )
        hard_negative_ctxs = (
            json_sample["hard_negative_ctxs"]
            if "hard_negative_ctxs" in json_sample
            else []
        )

        for ctx in positive_ctxs + negative_ctxs + hard_negative_ctxs:
            if "title" not in ctx:
                ctx["title"] = None

        def create_passage(ctx: dict):
            return BiEncoderPassage(
                normalize_passage(ctx["text"]) if self.normalize else ctx["text"],
                ctx["title"],
            )

        r.positive_passages = [create_passage(ctx) for ctx in positive_ctxs]
        r.negative_passages = [create_passage(ctx) for ctx in negative_ctxs]
        r.hard_negative_passages = [create_passage(ctx) for ctx in hard_negative_ctxs]
        r.img_id = json_sample['img_id']
        return r

    def __len__(self):
        return len(self.data)

    def get_qas(self) -> Tuple[List[str], List[str]]:
        return [s["question"] for s in self.data], [s["answers"] for s in self.data]

    def get_qas_range(
        self, start_idx: int, end_idx: int
    ) -> Tuple[List[str], List[str]]:
        return (
            [s["question"] for s in self.data[start_idx:end_idx]],
            [s["answers"] for s in self.data[start_idx:end_idx]],
        )

class RetrieverData(torch.utils.data.Dataset):
    def __init__(self, file: str):
        """
        :param file: - real file name or the resource name as they are defined in download_data.py
        """
        self.file = file
        self.data_files = []

    def load_data(self):
        self.data_files = [self.file]
        assert (
            len(self.data_files) == 1
        ), "RetrieverData source currently works with single files only. Files specified: {}".format(
            self.data_files
        )

        
class CsvCtxSrc(RetrieverData):
    def __init__(
        self,
        file: str,
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        super().__init__(file)
        self.text_col = text_col
        self.title_col = title_col
        self.id_col = id_col
        self.id_prefix = id_prefix
        self.normalize = normalize

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):
        super().load_data()
        if not os.path.exists(self.file):
            self.file = os.path.join(PROJ_PATH, self.file)
        with open(self.file) as ifile:
            reader = csv.reader(ifile, delimiter=",")
            for row in reader:
                if row[self.id_col] == "kid":
                    continue
                if self.id_prefix:
                    sample_id = self.id_prefix + str(row[self.id_col])
                else:
                    sample_id = row[self.id_col]
                passage = row[self.text_col]
                if self.normalize:
                    passage = normalize_passage(passage)
                ctxs[sample_id] = BiEncoderPassage(passage, '')

