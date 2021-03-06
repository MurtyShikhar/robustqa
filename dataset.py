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

def get_topic_id_pair(save_dir, orig_source=False, kmeans=False):
    # a unique <topic:id> mapping per process
    orig_main_sources = ['squad', 'newsqa', 'nat_questions', 'duorc', 'race', 'relation_extraction']
    if orig_source:
        topic_id_pair = {element:idx for idx, element in enumerate(orig_main_sources)} 
        topic_id_file = None
    else:
        # neither orig_source or kmeans is True
        # use the topics in the files
        topic_id_file = f'{save_dir}/topic_id_pair_{UUID}'
        if os.path.exists(topic_id_file):
            topic_id_pair = json.loads(open(topic_id_file).read())
        else:
            topic_id_pair = {}
    return topic_id_file, topic_id_pair

def save_topic_id_pair(topic_id_file, topic_id_pair):
    if topic_id_file is not None:
        with open(topic_id_file, 'w') as f:
            json.dump(topic_id_pair, f)

def get_topic_id(group, topic_id_pair, orig_source=False, kmeans=False):
    if "topic" in group:
        # only training data has topic
        # all training data has topics
        topic = group["topic"]
        if topic not in topic_id_pair:
            if orig_source:
                # treat unknown topics (outside of the main sources) as "squad"
                topic = "squad"
            else:
                # neither orig_source or kmeans is True
                # use the topics in the files
                new_id = len(topic_id_pair)
                topic_id_pair[topic] = new_id
        return topic, topic_id_pair[topic], topic_id_pair
    else:
        return None, -1, topic_id_pair

def add_question_to_dict(qa, context, topic, topic_id, data_dict):
    question = qa['question']
    if len(qa['answers']) == 0:
        data_dict['question'].append(question)
        data_dict['context'].append(context)
        data_dict['id'].append(qa['id'])
        data_dict['topic'].append(topic)
        data_dict['topic_id'].append(topic_id)
    else:
        for answer in qa['answers']:
            data_dict['question'].append(question)
            data_dict['context'].append(context)
            data_dict['id'].append(qa['id'])
            data_dict['answer'].append(answer)
            data_dict['topic'].append(topic)
            data_dict['topic_id'].append(topic_id)

def collapse_data_dict(data_dict):
    id_map = ddict(list)
    for idx, qid in enumerate(data_dict['id']):
        id_map[qid].append(idx)

    data_dict_collapsed = {'question': [], 'context': [], 'id': [], 'topic': [], 'topic_id': []}
    if data_dict['answer']:
        data_dict_collapsed['answer'] = []
    for qid in id_map:
        ex_ids = id_map[qid]
        data_dict_collapsed['question'].append(data_dict['question'][ex_ids[0]])
        data_dict_collapsed['context'].append(data_dict['context'][ex_ids[0]])
        data_dict_collapsed['topic'].append(data_dict['topic'][ex_ids[0]])
        data_dict_collapsed['topic_id'].append(data_dict['topic_id'][ex_ids[0]])
        data_dict_collapsed['id'].append(qid)
        if data_dict['answer']:
            all_answers = [data_dict['answer'][idx] for idx in ex_ids]
            data_dict_collapsed['answer'].append({
                'answer_start': [answer['answer_start'] for answer in all_answers],
                'text': [answer['text'] for answer in all_answers]
            })
    
    return data_dict_collapsed

def read_squad(path, save_dir, orig_source=False, kmeans=False):
    # parameters:
    # path: path of the file to read from
    # save_dir: the dir to save the topic_id_pair file where uniq
    #           topic IDs from the topics are stored
    # orig_source: Flag for whether use the original source as
    #              topic IDs, namely "squad", "newsqa", "nat_questions",
    #              "duorc", "race", "relation_extraction"
    # kmeans: Use the kmeans clusters as topic IDs.

    # only used when orig_source is True

    path = Path(path)

    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    topic_id_file, topic_id_pair = get_topic_id_pair(save_dir, orig_source, kmeans)

    data_dict = {'question': [], 'context': [], 'id': [], 'answer': [], 'topic': [], 'topic_id': []}
    for group in squad_dict['data']:
        topic, topic_id, topic_id_pair = get_topic_id(group, topic_id_pair, orig_source, kmeans)
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                add_question_to_dict(qa, context, topic, topic_id, data_dict)

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
