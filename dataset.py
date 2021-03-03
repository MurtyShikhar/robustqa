import torch
from pathlib import Path
from torch.utils.data import Dataset
from collections import Counter, OrderedDict, defaultdict as ddict

import os
import uuid
import json

UUID = str(uuid.uuid1()) + str(os.getpid())

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

def merge(encodings, new_encoding):
    if not encodings:
        return new_encoding
    else:
        for key in new_encoding:
            encodings[key] += new_encoding[key]
        return encodings

def get_topic_id_pair(save_dir, uuid):
    # a unique <topic:id> mapping per process
    topic_id_file = f'{save_dir}/topic_id_pair_{UUID}'
    if os.path.exists(topic_id_file):
        return topic_id_file, json.loads(open(topic_id_file).read())
    else:
        return topic_id_file, {}

def save_topic_id_pair(topic_id_file, topic_id_pair):
    with open(topic_id_file, 'w') as f:
        json.dump(topic_id_pair, f)

def get_topic_id(group, topic_id_pair):
    if "topic" in group:
        # only training data has topic
        topic = group["topic"]
        new_id = len(topic_id_pair)
        if topic not in topic_id_pair:
            topic_id_pair[topic] = new_id
        return topic_id_pair[topic]
    else:
        return -1

def add_question_to_dict(qa, context, topic_id, data_dict):
    question = qa['question']
    if len(qa['answers']) == 0:
        data_dict['question'].append(question)
        data_dict['context'].append(context)
        data_dict['id'].append(qa['id'])
        data_dict['topic_id'].append(topic_id)
    else:
        for answer in qa['answers']:
            data_dict['question'].append(question)
            data_dict['context'].append(context)
            data_dict['id'].append(qa['id'])
            data_dict['answer'].append(answer)
            data_dict['topic_id'].append(topic_id)

def collapse_data_dict(data_dict):
    id_map = ddict(list)
    for idx, qid in enumerate(data_dict['id']):
        id_map[qid].append(idx)

    data_dict_collapsed = {'question': [], 'context': [], 'id': [], 'topic_id': []}
    if data_dict['answer']:
        data_dict_collapsed['answer'] = []
    for qid in id_map:
        ex_ids = id_map[qid]
        data_dict_collapsed['question'].append(data_dict['question'][ex_ids[0]])
        data_dict_collapsed['context'].append(data_dict['context'][ex_ids[0]])
        data_dict_collapsed['topic_id'].append(data_dict['topic_id'][ex_ids[0]])
        data_dict_collapsed['id'].append(qid)
        if data_dict['answer']:
            all_answers = [data_dict['answer'][idx] for idx in ex_ids]
            data_dict_collapsed['answer'].append({
                'answer_start': [answer['answer_start'] for answer in all_answers],
                'text': [answer['text'] for answer in all_answers]
            })
    
    return data_dict_collapsed

def read_squad(path, save_dir):
    path = Path(path)
    topic_id_file, topic_id_pair = get_topic_id_pair(save_dir, uuid)

    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    data_dict = {'question': [], 'context': [], 'id': [], 'answer': [], 'topic_id': []}
    for group in squad_dict['data']:
        topic_id = get_topic_id(group, topic_id_pair)
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                add_question_to_dict(qa, context, topic_id, data_dict)

    data_dict_collapsed = collapse_data_dict(data_dict)
    save_topic_id_pair(topic_id_file, topic_id_pair)

    return data_dict_collapsed

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