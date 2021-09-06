# Visual Dense Passage Retrieval (Vis-DPR) 

This repo implements the idea in [Weakly-Supervised Visual-Retriever-Reader for Knowledge-based Question Answering](link). The visual retriever is built upon [Dense Passage Retrieveal(DPR)](https://github.com/facebookresearch/DPR) with new visual features and aim to retrieve knowledge to OKVQA (cite), a knowledge-based visual question answering benchmark. The visual reader is adapted from [hugginface](https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py) question answering example. 

## Features
1. Caption-based retrieval: the caption is added after to the question.
2. Image-based retrieval: the question encoder is based on LXMERT (cite), where a cross-representation of question and image is used as the question representation as original DPR.
3. Visual extractive reader: the extractive reader based on RoBERTA (cite), where sepecial word "unanswerable" and a caption of an image is added in front of the context. 


If you find this paper or this code useful, please cite this paper:

```
```

## Installation

Installation from the source. Python's virtual or Conda environments are recommended.

```bash
git clone https://github.com/luomancs/retriever_reader_for_okvqa.git
cd retriever_reader_for_okvqa
pip install -r requirements.txt
```

Visual-DPR is tested on Python 3.7 and PyTorch 1.7.1.


## Resources & Data formats
Corpus

We provide four types of corpus
1. okvqa_train_corpus: the corpus is collected based on the training data. corpus size 112,724
  
2. okvqa_full_corpus: the corpus is collected based on the training data and testing data 168,306
  
3. okvqa_train_clean_corpus: the corpus is based on okvqa_train_corpus but filtered with similar process as [T5] (https://arxiv.org/pdf/1910.10683.pdf), detailed process referred to paper. corpus size 111,412
  
4. okvqa_full_clean_corpus: the corpus is based on okvqa_full_corpus with same cleannp method as corpus 3. corpus size 166,390


Training data: you need to prepare data for either retriever or reader training.
1. for caption-retriever: download the training data from [google drive] (link). The training data is from OKVQA, where we use [OSCAR] (https://github.com/microsoft/Oscar) to generate the caption for the corresponding image. 
2. for image-retriever: download the image features from [google drive] (link). The image features is extracted using [Mask-RCNN] (https://github.com/facebookresearch/maskrcnn-benchmark)
3. for reader: download the training data from [google drive](link). The training data includes the question from OKVQA and the knowledge from the corpus.


## Retriever training

Coming soon

## Retriever inference

### Generating representation vectors for entire corpus.

```bash
python DPR/generate_dense_embeddings.py \
	model_file={path to biencoder checkpoint} \
	ctx_src={name of the passages resource, set to dpr_wiki to use our original wikipedia split} \
	shard_id={shard_num, 0-based} num_shards={total number of shards} \
	out_file={result files location + name PREFX}	
	encoder=hf_bert   
```

ctx_src: one of the corpus name (see DPR/conf/ctx_sources/okvqa_sources.yaml file).
encoder: either hf_bert (caption-dpr) or hf_lxmert_bert (image-dpr)

You can download already generated corpus embeddings from our original caption-dpr model from [google_drive](link)


### Retriever evaluation against the entire set of documents:

#### Retriever knowledge by Caption-DPR

```bash

python DPR/caption_dense_retriever.py \
	model_file={path to biencoder checkpoint} \
	qa_dataset=okvqa_test \
	ctx_datatsets=[{list of passage sources's names, comma separated without spaces}] \
	encoded_ctx_files=[{list of encoded document files glob expression, comma separated without spaces}] \
	out_file={path to output json file with results} 
	
```

#### Retriever knowledge by Caption-DPR

```bash

python DPR/image_dense_retriever.py \
	model_file={path to biencoder checkpoint} \
	qa_dataset=okvqa_test \
	ctx_datatsets=[{list of passage sources's names, comma separated without spaces}] \
	encoded_ctx_files=[{list of encoded document files glob expression, comma separated without spaces}] \
	out_file={path to output json file with results} 
	
```


## EReader model training

Comming soon

## EReader model inference

```bash
python evaluation/predict_answer.py \
--model_path {path to the EReader, you can download our pretrained model from [google drive]} \
--retrieve_kn_file {path to the retrieved knowledge given by the retriever} \
--prediction_save_path {path to save the prediction} \
--cuda_id 0 {-1 if evaluate on cpu}\
```

## CReader model training

Comming soon

## CReader model inference

Comming soon
