import pandas as pd
from torch import mps
from transformers import BertTokenizer, BertForTokenClassification
import os
from data_processor import Data_Processor
from dataset import MakeDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import torch.optim as optim
import warnings

warnings.filterwarnings("ignore")

FILE_NAME = 'ner_data.csv'
TRAIN_SIZE = .8
ID = 'bert-base-uncased'
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10

DEVICE = 'mps' if mps.is_available() else 'cpu'

train_params = {
    'batch_size' : TRAIN_BATCH_SIZE,
    'shuffle' : True,
    'num_workers' : 0
}

test_params = {
    'batch_size' : VALID_BATCH_SIZE,
    'shuffle' : True,
    'num_workers' : 0
}

def load_data(file_name):
    path = os.path.join("bert-ner/data", file_name)

    if os.path.exists(path):
        print("data Loaded..")
        data = pd.read_csv(path, encoding= 'unicode_escape')
        print(f"data shape: {data.shape}")
    else:
        print("error: data does not exists!!")
    
    return data

def prepare_data(data):
    print('processing start..')
    processor = Data_Processor(data = data)
    data, label2id, id2label = processor.process()

    print(f"processed data shape: {data.shape}")
    print(f"label2id length: len {len(label2id)}")
    print(f"id2label length: len {len(id2label)}")

    print(f"labe2id:{label2id}")
    print(f"id2label:{id2label}")

    print('processing done..')

    return data, label2id, id2label

def train_test_split(data, train_size):
    train_data = data.sample(frac = train_size, random_state = 200)
    test_data = data.drop(train_data.index).reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)

    return train_data, test_data


if __name__ == "__main__":

    data = load_data(FILE_NAME)
    data, label2id, id2label = prepare_data(data)
    train_data, test_data = train_test_split(data, train_size=TRAIN_SIZE)

    tokenizer = BertTokenizer.from_pretrained(ID)

    train_dataset = MakeDataset(train_data, label2id, tokenizer, MAX_LENGTH)
    test_dataset = MakeDataset(test_data, label2id, tokenizer, MAX_LENGTH)

    training_loader = DataLoader(train_dataset, **train_params)
    testing_loader = DataLoader(test_dataset, **test_params)


    model = BertForTokenClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels=len(label2id), 
            id2label=id2label, 
            label2id=label2id
         )
    model.to(DEVICE)

    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    trainer  = Trainer(
        model=model,
        training_loader=training_loader,
        optimizer=optimizer,
        DEVICE=DEVICE,
        MAX_GRAD_NORM=MAX_GRAD_NORM
    )
    
    trainer.fit(EPOCHS)

    # save model & tokenizer

    path = '/Users/shakibibnashameem/Documents/Practice/bert/bert-ner/artifacts/'

    model.save_pretrained(path+'bert_trained')
    tokenizer.save_pretrained(path+'tokenizer')
