# coding=utf-8
# email: wangzejunscut@126.com

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def load_dataset(data_path, tokenizer, sep="\t", max_seq_length=1024):
    texts, labels = [], []
    with open(data_path, mode="r", encoding="utf-8") as handle:
        for line in tqdm(handle):
            line = line.rstrip().split(sep, maxsplit=1)
            if len(line) < 2:
                continue
            texts.append(line[1])
            labels.append(int(line[0]))

    encodings = tokenizer(
        texts, 
        truncation=True,
        add_special_tokens=False, 
        max_length=max_seq_length, 
        return_token_type_ids=False
    )
    return (encodings, labels)
    

class ClsDataset(Dataset):
    def __init__(self, data_path, tokenizer, sep="\t", max_seq_length=1024):
        super(ClsDataset, self).__init__()
        self.data = load_dataset(data_path, tokenizer, sep, max_seq_length)
        
    def __len__(self):
        """
        Returns the number of samples in dataset.
        """
        return len(self.data[1])
    
    def __getitem__(self, index):
        """
        Basic function of `ClsDataset` to get sample from dataset with a given index.
        """
        return (self.data[0]["input_ids"][index], self.data[0]["attention_mask"][index], self.data[1][index])

class ClsCollator:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        input_ids = [item[0] for item in batch]
        attention_mask = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        
        # Padding
        max_seq_len = max(len(_) for _ in input_ids)
        for i in range(len(batch)):
            pad_num = max_seq_len - len(input_ids[i])
            input_ids[i].extend([self.pad_token_id] * pad_num)
            attention_mask[i].extend([0] * pad_num)
        
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        labels = torch.tensor(labels)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

