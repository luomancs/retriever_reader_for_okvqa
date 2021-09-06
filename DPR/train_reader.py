from datasets import load_dataset

import argparse
import transformers



from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering, 
    TrainingArguments, 
    Trainer,
    default_data_collator
)

def prepare_train_features(examples, args):
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["title" if args.pad_on_right else "text"],
        examples["text" if args.pad_on_right else "title"],
        truncation="only_second" if args.pad_on_right else "only_first",
        max_length=args.max_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answer = examples["answer"][sample_index]
        # If no answers are given, set the cls_index as answer.
        try:
            # Start/end character index of the answer in the text.
            context = examples['text'][sample_index]
            start_char = context.lower().index(answer.lower())
            end_char = start_char + len(answer)

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
            
        except:
        
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        

    return tokenized_examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_data_files', type=str, required = True, default=None,
                        help="path to the training data")
    
    parser.add_argument('--testdev_data_files', type=str, required = True, default=None,
                        help="path to the testdev data")
    
    parser.add_argument('--pretrained_model', type=str, default="deepset/roberta-large-squad2",
                        help="path to the pretrained model")
    
    parser.add_argument('--save_model_path', type=str, required = True, default=None,
                        help="path to the save model")
    
    parser.add_argument('--max_length', type=int, default=384,
                        help="the maximum length of a feature (question and context)")
    
    parser.add_argument('--doc_stride', type=int, default=None,
                        help="the authorized overlap between two part of the context when splitting it is needed.")
    
    parser.add_argument('--num_train_epochs', type=int, default=3,
                        help="num of training epochs")
    parser.add_argument('--save_steps', type=int, default=10000,
                        help="num of step to save a checkpoint ")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help="gradient accumulation steps")
    parser.add_argument('--per_device_train_batch_size', type=int, default=4,
                        help="training batch size")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=32,
                        help="evaluation batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help="learning rate")
    
    
    
    args = parser.parse_args()

    

    model_checkpoint = args.pretrained_model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

    max_length = args.max_length 
    doc_stride = args.doc_stride 

    args.pad_on_right = tokenizer.padding_side == "right"
    
    dataset = load_dataset('csv', data_files=[args.train_data_files])
    test_data = load_dataset('csv', data_files=[args.testdev_data_files])

    tokenized_datasets = dataset.map(lambda example: prepare_train_features(example, args), batched=True)
    tokenized_datasets_test = test_data.map(lambda example: prepare_train_features(example, args), batched=True)


    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

    args = TrainingArguments(
        args.save_model_path,
        evaluation_strategy = "epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        save_steps= args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )


    data_collator = default_data_collator

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset = tokenized_datasets_test['train'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model(args.save_model_path)
