import util
import numpy as np
from nltk import tokenize
# conda install spacy
# python -m spacy download en_core_web_sm
import spacy

# load the module
nlp = spacy.load('en_core_web_sm')

def sample_dataset(args, datasets, data_dir, sample_prob = 0.1, seed = 94305,
                   sample_queries_dir = 'queries/sample_queries.txt',
                   sample_context_dir = 'queries/sample_context.txt'):
    np.random.seed(seed)
    datasets = datasets.split(',')
    dataset_dict = None
    dataset_name=''
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
    train_length = len(dataset_dict['id'])
    sample_idx = list(np.random.choice(train_length, size = int(sample_prob * train_length), replace = False))

    sample_queries = [dataset_dict['question'][i] for i in sample_idx]
    sample_context = [dataset_dict['context'][i] for i in sample_idx]
    gold_answers = [dataset_dict['answer'][i] for i in sample_idx]
    
    write_queries(sample_queries, sample_queries_dir)
    sample_context_individual_length, answer_locs = write_context(sample_context, gold_answers, sample_context_dir)
        
    return dataset_dict, sample_idx, sample_context_individual_length, gold_answers, answer_locs
    

def write_queries(queries, output_dir = 'queries/sample_queries.txt'):
    with open(output_dir, 'w') as f:
        for q in queries:
            f.write(q + '\n')

def write_context(context, gold_answers, output_dir = 'queries/sample_context.txt'):
    out_lengths = []
    answer_locs = []
    with open(output_dir, 'w') as f:
        for i in range(len(context)):
            out = [str(sent).strip() for sent in nlp(context[i].replace('\n', '')).sents if str(sent).strip() != '']
            for j in range(len(out)):
                f.write(out[j] + '\n')
                if gold_answers[i]['text'] in out[j]:
                  answer_locs.append(j)
            out_lengths.append(len(out))
    return out_lengths, answer_locs

def concat_queries(queries_dir):
    output_queries = []
    f = open(queries_dir, 'r')
    whole_queries = f.readlines()
    for q in whole_queries:
        output_queries.append(q)
    return output_queries
    
def concat_context(context_dir, sample_context_individual_length):
    output_context = []
    count = 0
    f = open(context_dir, 'r')
    whole_context = f.readlines()
    for l in sample_context_individual_length:
        individual_context = whole_context[count:(count+l)]
        individual_context = [ic.rstrip() for ic in individual_context]
        individual_context = ' '.join(individual_context)
        output_context.append(individual_context)
        count += l
    return output_context
  
def get_empty_trans_index(queries_dir, context_dir, sample_context_individual_length,
                      output_queries_dir, output_context_dir):
    q_file = open(queries_dir, 'r')
    c_file = open(context_dir, 'r')
    output_q_file = open(output_queries_dir, 'w')
    output_c_file = open(output_context_dir, 'w')
    
    num_samples = len(sample_context_individual_length)
    drop_index = []
    
    for i in range(num_samples):
      drop = False
      q = q_file.readline()
      
      if q == '\n':
        drop = True
      else:
        context = []
        for j in range(sample_context_individual_length[i]):
          c = c_file.readline()
          if c == '\n':
            drop = True
            break
          else:
            context.append[c]
      
      if drop:
        drop_index.append(i)
      else:
        output_q_file.write(q)
        for c in context:
          output_c_file.write(c)
          
    return drop_index
  
def drop_empty_trans(queries_dir, context_dir, sample_context_individual_length,
                      output_queries_dir, output_context_dir, process_lists):
    drop_index = get_empty_trans_index(queries_dir, context_dir, sample_context_individual_length,
                                       output_queries_dir, output_context_dir)

    for l in process_lists:
      l = [elem for idx, elem in enumerate(l) if idx not in drop_index]
    
    return process_lists
