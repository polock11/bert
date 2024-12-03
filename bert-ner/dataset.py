import torch
from torch.utils.data import Dataset

class MakeDataset(Dataset):
    def __init__(self, data, label2id, tokenizer, MAX_LENGTH):
        self.data = data
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_length = MAX_LENGTH

        # Pre-tokenize and prepare data
        self.tokenized_data = self._prepare_data()

    def _prepare_data(self):
        tokenized_data = []

        for sentence, word_labels in zip(self.data["Sentence"], self.data["Word_labels"]):
            # Tokenize sentence
            encoded_dict = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens = True,
                max_length = self.max_length,
                pad_to_max_length = True,
                return_attention_mask = True,
                return_tensors = 'pt'
            )

            # Prepare labels
            labels = word_labels.split(' ')
            label_ids = [self.label2id[label] for label in labels]
            label_ids = [-100] + label_ids + [-100]  # Add special token labels
            label_ids += [-100] * (self.max_length - len(label_ids))  # Pad labels to max_length
            label_ids = label_ids[:self.max_length]  # Truncate if necessary

            # Add to tokenized data
            tokenized_data.append({
                "input_ids": encoded_dict['input_ids'].squeeze(0),
                "attention_mask": encoded_dict['attention_mask'].squeeze(0),
                "labels": torch.tensor(label_ids)
            })

        return tokenized_data

    def __getitem__(self, index):
        return self.tokenized_data[index]

    def __len__(self):
        return len(self.data)
