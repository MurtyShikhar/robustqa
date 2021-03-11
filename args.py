import argparse

def get_train_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num-visuals', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='save/')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--train-datasets', type=str, default='squad,nat_questions,newsqa')
    parser.add_argument('--run-name', type=str, default='multitask_distilbert')
    parser.add_argument('--recompute-features', action='store_true')
    parser.add_argument('--train-dir', type=str, default='datasets/indomain_train')
    parser.add_argument('--val-dir', type=str, default='datasets/indomain_val')
    parser.add_argument('--eval-dir', type=str, default='datasets/oodomain_test')
    parser.add_argument('--eval-datasets', type=str, default='race,relation_extraction,duorc')
    parser.add_argument('--do-train', action='store_true')
    parser.add_argument('--do-eval', action='store_true')
    parser.add_argument('--sub-file', type=str, default='')
    parser.add_argument('--visualize-predictions', action='store_true')
    parser.add_argument('--eval-every', type=int, default=5000)
    args = parser.parse_args()
    return args
    
def get_nmt_args(beam=1):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-datasets', type=str, default='squad,nat_questions,newsqa')
    parser.add_argument('--train-dir', type=str, default='datasets/indomain_train')
    parser.add_argument('--sample_prob', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
        
    # where to save the sampling (with sampling prob) queries and context
    parser.add_argument('--sample_queries_dir', type=str, default='2_layer_nmt/QA/sample_queries.txt')
    parser.add_argument('--sample_context_dir', type=str, default='2_layer_nmt/QA/sample_context.txt')
    
    # where to retrieve the translated queries and context
    parser.add_argument('--trans_queries_dir', type=str, default='2_layer_nmt/QA/trans_en_es_queries_beam_{0}.txt'.format(beam))
    parser.add_argument('--trans_context_dir', type=str, default='2_layer_nmt/QA/trans_en_es_context_beam_{0}.txt'.format(beam))
    
    # where to store the blank line dropped translated queries and context
    parser.add_argument('--dropped_queries_dir', type=str, default='2_layer_nmt/QA/trans_en_es_queries_beam_{0}_dropped.txt'.format(beam))
    parser.add_argument('--dropped_context_dir', type=str, default='2_layer_nmt/QA/trans_en_es_context_beam_{0}_dropped.txt'.format(beam))

     # where to retrieve the back translated queries and context
    parser.add_argument('--back_trans_queries_dir', type=str, default='2_layer_nmt/QA/trans_es_en_queries_beam_{0}.txt'.format(beam))
    parser.add_argument('--back_trans_context_dir', type=str, default='2_layer_nmt/QA/trans_es_en_context_beam_{0}.txt'.format(beam))
    
    # where to store the blank line dropped back translated queries and context
    parser.add_argument('--back_dropped_queries_dir', type=str, default='2_layer_nmt/QA/trans_es_en_queries_beam_{0}_dropped.txt'.format(beam))
    parser.add_argument('--back_dropped_context_dir', type=str, default='2_layer_nmt/QA/trans_es_en_context_beam_{0}_dropped.txt'.format(beam))
    # for sanity check bleu score
    parser.add_argument('--sample_context_dropped_dir', type=str, default='2_layer_nmt/QA/sample_context_beam_{0}_dropped.txt'.format(beam))
    
    # where to store augmented dataset
    parser.add_argument('--aug_dataset_dict', type=str, default='augmented_dataset_beam_{0}.json'.format(beam))
    parser.add_argument('--aug_dataset_pickle', type=str, default='augmented_dataset_beam_{0}.pickle'.format(beam))
    args = parser.parse_args()
    return args
