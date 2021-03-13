import util
import numpy as np
from nltk import tokenize
# conda install spacy
# python -m spacy download en_core_web_sm
import spacy
import sacrebleu
import pickle
import json
from py_stringmatching import GeneralizedJaccard

# load the module
nlp = spacy.load('en_core_web_sm')


def sample_dataset(args, datasets, data_dir, sample_queries_dir, sample_context_dir,
                   sample_prob=0.1, seed=94305, max_num=150000):
    np.random.seed(seed)
    datasets = datasets.split(',')
    dataset_dict = None
    dataset_name = ''
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
    train_length = len(dataset_dict['id'])
    sample_idx = list(np.random.choice(train_length, size=int(sample_prob * train_length), replace=False))[:max_num]

    sample_queries = [dataset_dict['question'][i] for i in sample_idx]
    sample_context = [dataset_dict['context'][i] for i in sample_idx]
    gold_answers = [dataset_dict['answer'][i] for i in sample_idx]

    write_queries(sample_queries, sample_queries_dir)
    sample_context_individual_length, answer_locs = write_context(sample_context, gold_answers, sample_context_dir)

    print('Sampled queries saved at:', sample_queries_dir)
    print('Sampled context saved at:', sample_context_dir)
    print('Num of examples sampled:', len(sample_idx))

    return sample_idx, sample_context_individual_length, gold_answers, answer_locs


def write_queries(queries, output_dir):
    with open(output_dir, 'w') as f:
        for q in queries:
            if not q.endswith('?'):
                q += '?'
            f.write(q + '\n')

    f.close()


def write_context(context, gold_answers, output_dir):
    out_lengths = []
    answer_locs = []

    f = open(output_dir, 'w')

    for i in range(len(context)):
        out = [(str(sent).encode('ascii', 'ignore')).decode("utf-8").strip() for sent in
               nlp(context[i].replace('\n', '')).sents if
               (str(sent).encode('ascii', 'ignore')).decode("utf-8").strip() != '']

        curr_answers = gold_answers[i]['text']
        curr_locs = [-1] * len(curr_answers)

        for j in range(len(out)):
            f.write(out[j] + '\n')
            for k in range(len(curr_answers)):
                if curr_answers[k] in out[j]:
                    curr_locs[k] = j

        out_lengths.append(len(out))
        answer_locs.append(curr_locs)

    f.close()
    return out_lengths, answer_locs


def get_keep_index(queries_dir, context_dir, sample_context_individual_length,
                   output_queries_dir, output_context_dir):
    q_file = open(queries_dir, 'r')
    c_file = open(context_dir, 'r')
    output_q_file = open(output_queries_dir, 'w')
    output_c_file = open(output_context_dir, 'w')

    num_samples = len(sample_context_individual_length)
    keep_index = []

    for i in range(num_samples):
        drop = False
        q = q_file.readline()

        if q.strip() == '':
            drop = True
        else:
            context = []
            for j in range(sample_context_individual_length[i]):
                c = c_file.readline()
                if c.strip() == '':
                    drop = True
                else:
                    context.append(c)

        if not drop:
            keep_index.append(i)
            output_q_file.write(q)
            for c in context:
                output_c_file.write(c)

    q_file.close()
    c_file.close()
    output_q_file.close()
    output_c_file.close()

    return keep_index


def clean_lists(keep_index, process_lists):
    cleaned_lists = []
    for l in process_lists:
        cleaned_lists.append([elem for idx, elem in enumerate(l) if idx in keep_index])
    return cleaned_lists


def compute_answer_span(context_sent, gold_answer, sim_measure=GeneralizedJaccard):
    """
        input:
            context_sent <string>: the paraphrased sentence which contains the answer before back-translation
            gold_answer <string>: the answer phrase
            similarity measure: by default we use generalized jaccard similarity which gives stable performance under misspelling
        returns:
            start_pos <int>: stores start index of the estimated answer span in this context sentence
            target_substring <string>: the estimated answer span
    """

    context_sent_token = [token.text_with_ws for token in nlp(context_sent)]
    answer_sent_token = [token.text_with_ws for token in nlp(gold_answer)]
    n = len(context_sent_token)

    me = sim_measure()
    best_jac_score = float('-inf')
    best_substring = ''

    for i in range(n):
        for j in range(n - i):
            current_score = me.get_raw_score(context_sent_token[i:n - j], answer_sent_token)
            if current_score > best_jac_score:
                best_jac_score = current_score
                best_substring = ''.join(context_sent_token[i:n - j]).strip()

    start_pos = context_sent.find(best_substring)
    return start_pos, best_substring, best_jac_score


def get_trans_context_answers(context_dir, sample_context_individual_length,
                              gold_answers, answer_locs, threshold):
    """
        input:
            context_dir <file>: the back translated context file
            sample_context_individual_length list<integer>: number of sentences in each context
            gold_answers list<list<string>>: the list of gold answers
            answer_locs list<list<integer>>: the list of answer locs
        returns:
            new_answers: list of new_answers
    """
    in_file = open(context_dir, 'r')

    num_samples = len(sample_context_individual_length)
    keep_index = []
    new_answers = []

    for i in range(num_samples):
        curr_answers = gold_answers[i]['text']
        curr_locs = answer_locs[i]

        new_start_idx = []
        new_curr_answers = []
        jac_scores = []
        char_count = 0

        for j in range(sample_context_individual_length[i]):
            context_sent = in_file.readline().strip()

            for k in range(len(curr_locs)):
                if j == curr_locs[k]:
                    start_pos, best_substring, best_jac_score = compute_answer_span(context_sent, curr_answers[k])
                    new_start_idx.append(char_count + start_pos)
                    new_curr_answers.append(best_substring)
                    jac_scores.append(best_jac_score)

            char_count += len(context_sent + " ")

        if len(jac_scores) > 0 and max(jac_scores) > threshold:
            keep_index.append(i)
            new_answers.append(dict({'answer_start': new_start_idx, 'text': new_curr_answers}))

    in_file.close()
    return keep_index, new_answers


def concat(file_dir):
    with open(file_dir, 'r') as f:
        output = [line.strip() for line in f]
    return output


def concat_context(context_dir, sample_context_individual_length):
    output_context = []
    count = 0
    f = open(context_dir, 'r')
    whole_context = f.readlines()
    for l in sample_context_individual_length:
        individual_context = whole_context[count:(count + l)]
        individual_context = [ic.strip() for ic in individual_context]
        individual_context = ' '.join(individual_context)
        output_context.append(individual_context)
        count += l
    f.close()
    return output_context


def drop_files(keep_index, queries_dir, context_dir, dropped_queries_dir, dropped_context_dir,
               sample_context_individual_length):
    q_file = open(queries_dir, 'r')
    c_file = open(context_dir, 'r')
    output_q_file = open(dropped_queries_dir, 'w')
    output_c_file = open(dropped_context_dir, 'w')
    num_samples = len(sample_context_individual_length)

    print("In drop_files function, number of samples:" + str(num_samples))

    for i in range(num_samples):
        query = q_file.readline()
        if i in keep_index:
            output_q_file.write(query)

        for j in range(sample_context_individual_length[i]):
            context_sent = c_file.readline()
            if i in keep_index:
                output_c_file.write(context_sent)

    q_file.close()
    c_file.close()
    output_q_file.close()
    output_c_file.close()


def compute_backtrans_bleu(sample_queries_dir, sample_context_dir, backtrans_queries_dir, backtrans_context_dir):
    queries_bleu = sacrebleu.corpus_bleu(concat(backtrans_queries_dir), [concat(sample_queries_dir)])
    print('Queries back translation BLEU: {}'.format(queries_bleu.score))
    context_bleu = sacrebleu.corpus_bleu(concat(backtrans_context_dir), [concat(sample_context_dir)])
    print('Context back translation BLEU: {}'.format(context_bleu.score))


def gen_augmented_dataset(aug_data_name, backtrans_queries_dir, backtrans_context_dir,
                          sample_context_individual_length, sample_idx, new_answers):
    backtranslated_queries = concat(backtrans_queries_dir)
    backtranslated_context = concat_context(backtrans_context_dir, sample_context_individual_length)
    qids = ['aug' + aug_data_name + 'num' + str(x) for x in sample_idx]

    new_data_dict = {'question': [], 'context': [], 'id': [], 'answer': []}
    for question, context, qid, answer in zip(backtranslated_queries, backtranslated_context, qids, new_answers):
        new_data_dict['question'].append(question)
        new_data_dict['context'].append(context)
        new_data_dict['id'].append(qid)
        new_data_dict['answer'].append(answer)

    print('Num of backtranslated queries:', len(backtranslated_queries))
    print('Num of backtranslated context:', len(backtranslated_context))
    print('Num of augmented samples:', len(sample_idx))
    print('Num of new answers:', len(new_answers))

    print_augmented_dataset(new_data_dict)
    return new_data_dict


def print_augmented_dataset(new_data_dict):
    for i in range(10):
        print("========== Augmented example {0} ==========".format(i))
        print("question:", new_data_dict['question'][i])
        print("context:", new_data_dict['context'][i])
        print("id:", new_data_dict['id'][i])
        print("answer:", new_data_dict['answer'][i])


def save_as_pickle(new_data_dict, pickle_file):
    with open(pickle_file, 'wb') as f:
        pickle.dump(new_data_dict, f)
    print("Augmented data (pickle) saved at:", pickle_file)


def save_as_json(new_data_dict, json_file):
    with open(json_file, 'w', encoding='utf8') as f:
        json.dump(new_data_dict, f, indent=4)
    print("Augmented data (json) saved at:", json_file)
