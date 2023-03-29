# coding=utf-8
# email: wangzejunscut@126.com

import argparse
import numpy as np
from sklearn import metrics
from dataset import ClsDataset, ClsCollator
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    BertTokenizerFast, 
    GPT2ForSequenceClassification,
    Trainer,
    EvalPrediction
)

def compute_metrics(p: EvalPrediction):
    preds, labels = p
    preds = np.argmax(preds, axis=-1)
    acc = metrics.accuracy_score(labels, preds)
    precision = metrics.precision_score(labels, preds)
    recall = metrics.recall_score(labels, preds)
    f1 = metrics.f1_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def parse_args():
    parser = argparse.ArgumentParser(description="GPT2ForSequenceClassification")
    parser.add_argument("--train_args_file", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="uer/gpt2-chinese-cluecorpussmall")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    return args
    
def train(args):
    parser = HfArgumentParser(TrainingArguments)
    training_args, = parser.parse_json_file(json_file=args.train_args_file)
    training_args.local_rank = args.local_rank
        
    # set seed
    set_seed(training_args.seed)
    
    # load model and tokenizer
    model = GPT2ForSequenceClassification.from_pretrained(args.pretrained_model_name_or_path, num_labels=2)
    tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_model_name_or_path)
    model.config.pad_token_id = tokenizer.pad_token_id

    # load dataset
    train_dataset = ClsDataset(args.train_file, tokenizer, max_seq_length=args.max_seq_length)
    eval_dataset = ClsDataset(args.eval_file, tokenizer, max_seq_length=args.max_seq_length)
    data_collator = ClsCollator(pad_token_id=tokenizer.pad_token_id)
   
    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    trainer.train()

if __name__ == "__main__":
    args = parse_args()
    train(args)

