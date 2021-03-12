from args import get_nmt_args
from backtranslate_util import *
import util
import sacrebleu
# from transformers import DistilBertTokenizerFast

def nmt_sampling(args):
    # sampling
    dataset_dict, sample_idx, sample_context_individual_length, gold_answers, answer_locs = sample_dataset(args, args.train_datasets, args.train_dir,
                                                                                                       args.sample_queries_dir, args.sample_context_dir, 
                                                                                                       args.sample_prob, args.seed)


    print('Sampled queries are being saved at:', args.sample_queries_dir)         
    print('Sampled context are being saved at:', args.sample_context_dir)
    print('Num of examples sampled:', len(sample_idx))

    # forward translation
    keep_index_1 = get_keep_index(args.trans_queries_dir, args.trans_context_dir, sample_context_individual_length,
                              args.dropped_queries_dir, args.dropped_context_dir)
    sample_idx, dropped_context_individual_length, gold_answers, answer_locs = clean_lists(keep_index_1, [sample_idx, sample_context_individual_length, gold_answers, answer_locs])
    print('Num of non-empty examples after translation:', len(sample_idx))

    # back translation
    keep_index_2 = get_keep_index(args.back_trans_queries_dir, args.back_trans_context_dir, dropped_context_individual_length,
                              args.back_dropped_queries_dir, args.back_dropped_context_dir)
    sample_idx, dropped_context_individual_length, gold_answers, answer_locs = clean_lists(keep_index_2, [sample_idx, dropped_context_individual_length, gold_answers, answer_locs])
    print('Num of non-empty examples after back translation:', len(sample_idx))

    # estimate new answers
    new_answers = get_trans_context_answers(args.back_dropped_context_dir, dropped_context_individual_length, 
                                        gold_answers, answer_locs)

    # compute queries and context BLEU
    keep_index = [elem for idx, elem in enumerate(keep_index_1) if idx in keep_index_2]
    sample_queries, sample_context = clean_sample_files(keep_index, args.sample_queries_dir, args.sample_context_dir, args.sample_context_dropped_dir, sample_context_individual_length)
    queries_bleu = sacrebleu.corpus_bleu(concat(args.back_dropped_queries_dir), [sample_queries])
    print('Queries back translation BLEU: {}'.format(queries_bleu.score))
    context_bleu = sacrebleu.corpus_bleu(concat(args.back_dropped_context_dir), [sample_context])
    print('Context back translation BLEU: {}'.format(context_bleu.score))

    # create augmented dataset
    backtranslated_queries = concat(args.back_dropped_queries_dir)
    backtranslated_context = concat_context(args.back_dropped_context_dir, dropped_context_individual_length)
    qids = ['augbeam5num'+ str(x) for x in sample_idx]
    new_data_dict = gen_augmented_dataset(backtranslated_queries, backtranslated_context, qids, new_answers)
    
    # test
    print_augmented_dataset(new_data_dict)
    save_as_pickle(new_data_dict, args.aug_dataset_pickle)
    save_as_json(new_data_dict, args.aug_dataset_json)
    

# data_encodings = read_and_process(args, tokenizer, new_dataset_dict, data_dir, dataset_name, split_name)
# return util.QADataset(data_encodings, train=(split_name=='train')), dataset_dict

if __name__ == '__main__':
    args = get_nmt_args(beam=1)
    nmt_sampling(args)
    
#   args = get_nmt_args(beam=5)
#   nmt_sampling(args)

#   tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
#   output = get_sampling_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train')
