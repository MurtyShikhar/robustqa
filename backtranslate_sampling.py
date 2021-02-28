from args import get_train_test_args
import argparse
from backtranslate_util import sample_dataset, concat_context, concat_queries
from train import read_and_process
import util


def get_sampling_dataset(args, datasets, data_dir, tokenizer, split_name):
    # for testing purpose can de-function the code and uncomment the line below
    # args = get_train_test_args() 
    dataset_dict, sample_idx, sample_context_individual_length = sample_dataset(args, args.train_datasets, 
                                                                                args.train_dir,
                                                                                args.sample_prob, args.seed,
                                                                                args.sample_queries_dir,
                                                                                args.sample_context_dir)
    print('Sampled queries are being saved at:', args.sample_queries_dir)         
    print('Sampled context are being saved at:', args.sample_context_dir)                                                                      

    backtranslated_queries = concat_queries(args.backtranslate_queries_dir)
    backtranslated_context = concat_context(args.backtranslate_context_dir, sample_context_individual_length)

    new_dataset_dict = dict(dataset_dict)

    for (index, replacement) in zip(sample_idx, backtranslated_queries):
        new_dataset_dict['question'][index] = replacement

    for (index, replacement) in zip(sample_idx, backtranslated_context):
        new_dataset_dict['context'][index] = replacement

    # for testing purpose can comment out the two lines below and check new_dataset_dict
    data_encodings = read_and_process(args, tokenizer, new_dataset_dict, data_dir, dataset_name, split_name)
    return util.QADataset(data_encodings, train=(split_name=='train')), dataset_dict






    



