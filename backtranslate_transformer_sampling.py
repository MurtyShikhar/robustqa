from args import get_transformer_args
from backtranslate_util import *
import util
# from transformers import DistilBertTokenizerFast


args = get_transformer_args()
sample_idx, sample_context_individual_length, gold_answers, answer_locs = sample_dataset(args, args.train_datasets, args.train_dir,
                                                                                         args.sample_queries_dir, args.sample_context_dir, 
                                                                                         args.sample_prob, args.seed, max_num=10000)
