import torch
from torch.utils.data import Dataset
from collections import Counter, OrderedDict, defaultdict as ddict

def calculate_weights(Dataset):
    if 'topic_id' in Dataset:
        total = len(Dataset['topic_id'])
        topic_count = ddict(int)
        for topic in Dataset['topic_id']:
            topic_count[topic] += 1
        num_topics = len(topic_count)
        weights = [total/topic_count[i] for i in sorted(topic_count.keys())]
        return weights, num_topics
    else:
        return [], 0

class QADataset(Dataset):
    def __init__(self, encodings, train=True, evaluation=False, test=False):
        self.encodings = encodings
        self.keys = ['input_ids', 'attention_mask']
        if train:
            self.keys += ['topic_id', 'start_positions', 'end_positions']
            self.weights, self.num_topic = calculate_weights(encodings)
        elif evaluation:
            self.keys += ['start_positions', 'end_positions']
        assert(all(key in self.encodings for key in self.keys))

    def __getitem__(self, idx):
        return {key : torch.tensor(self.encodings[key][idx]) for key in self.keys}

    def __len__(self):
        return len(self.encodings['input_ids'])

    def topic_weights(self):
        return self.weights

    def num_topics(self):
        return self.num_topic