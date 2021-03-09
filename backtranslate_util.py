import util
import numpy as np
from nltk import tokenize
# conda install spacy
# python -m spacy download en_core_web_sm
import spacy
import sacrebleu
from py_stringmatching import GeneralizedJaccard


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
          if not q.endswith('?'):
            q += '?'
          f.write(q + '\n')

def write_context(context, gold_answers, output_dir = 'queries/sample_context.txt'):
    out_lengths = []
    answer_locs = []
    with open(output_dir, 'w') as f:
        for i in range(len(context)):
            out = [(str(sent).encode('ascii', 'ignore')).decode("utf-8").strip() for sent in nlp(context[i].replace('\n', '')).sents if (str(sent).encode('ascii', 'ignore')).decode("utf-8").strip() != '']
            curr_answers = gold_answers[i]['text']
            curr_locs = [-1] * len(curr_answers)

            for j in range(len(out)):
              f.write(out[j] + '\n')
              for k in range(len(curr_answers)):
                if curr_answers[k] in out[j]:
                  curr_locs[k] = j
                    
            out_lengths.append(len(out))
            answer_locs.append(curr_locs)
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

def concat(backtrans_file):
    output = []
    f = open(backtrans_file, 'r')
    contents = f.readlines()
    for elem in contents:
        output.append(elem)
    return output
 
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
            context.append(c)
      
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
    dropped_lists = []

    for l in process_lists:
      new_l = [elem for idx, elem in enumerate(l) if idx not in drop_index]
      dropped_lists.append(new_l)
    
    return dropped_lists

def compute_backtrans_bleu(original_file, backtrans_file):
  ref_file = open(original_file, 'r')
  pred_file = open(backtrans_file, 'r')
  
  refs = [line.strip() for line in ref_file]
  preds = [line.strip() for line in pred_file]
  bleu = sacrebleu.corpus_bleu(preds, [refs])
  return bleu.score

def compute_answer_span(context_sent, gold_answer, sim_measure = GeneralizedJaccard):
    """
        input:
            context_sent <string>: the paraphrased sentence which contains the answer before back-translation
            gold_answer <string>: the answer phrase
            similarity measure: by default we use generalized jaccard similarity which gives stable performance under misspelling
        returns:
            (start_pos, end_pos) <tuple>: stores start and end indices of the estimated answer span in this context sentence
            target_substring <string>: the estimated answer span
    """

    context_sent_token = [token.text_with_ws for token in nlp(context_sent)]
    answer_sent_token = [token.text_with_ws for token in nlp(gold_answer)]
    n = len(context_sent_token)

    me = sim_measure()
    best_jac_score = float('-inf')
    best_substring = ''
    
    for i in range(n):
        for j in range(n-i):
            substring = ''.join(context_sent_token[i:n-j])
            current_score = me.get_raw_score(context_sent_token[i:n-j], answer_sent_token)
            
            if current_score > best_jac_score:
                best_jac_score = current_score
                best_substring = substring

    start_pos = context_sent.find(best_substring)
    #end_pos = start_pos + len(best_substring)
    
    return start_pos, best_substring
  
def get_trans_context_answers(context_dir, sample_context_individual_length,
                              gold_answers, answer_locs, output_context_dir):
    """
        input:
            context_dir <file>: the back translated context file
            sample_context_individual_length list<integer>: number of sentences in each context
            gold_answers list<list<string>>: the list of gold answers
            answer_locs list<list<integer>>: the list of answer locs
            output_context_dir <file>: concatenated context
        returns:
            new_answers: list of new_answers
    """
    in_file = open(context_dir, 'r')
    out_file = open(output_context_dir, 'w')

    num_samples = len(sample_context_individual_length)
    new_answers = []

    for i in range(num_samples):
        curr_answers = gold_answers[i]['text']
        curr_locs = answer_locs[i]

        new_start_idx = []
        new_curr_answers = []
        curr_context = ''
        char_count = 0

        for j in range(sample_context_individual_length[i]):
            context_sent = in_file.readline().strip()

            for k in range(len(curr_locs)):
                if j == curr_locs[k]:
                    start_pos, best_substring = compute_answer_span(context_sent, curr_answers[k])
                    new_start_idx.append(char_count + start_pos)
                    new_curr_answers.append(best_substring)
            
            curr_context += context_sent + " "
            char_count += len(context_sent + " ")

        new_answers.append(dict({'answer_start': new_start_idx, 'text': new_curr_answers}))
        out_file.write(curr_context + '\n')

    return new_answers
