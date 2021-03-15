import argparse

def get_transformer_args(lang='de', indomain=True):
    if indomain: 
        parser = argparse.ArgumentParser()
        parser.add_argument('--queries_input_dir', type=str, default='QA/sample_queries.txt')
        parser.add_argument('--queries_trans_dir', type=str, default='QA/trans_en_{0}_queries.txt'.format(lang))
        parser.add_argument('--queries_backtrans_dir', type=str, default='QA/trans_{0}_en_queries.txt'.format(lang))
    
        parser.add_argument('--context_input_dir', type=str, default='QA/sample_context.txt')
        parser.add_argument('--context_trans_dir', type=str, default='QA/trans_en_{0}_context.txt'.format(lang))
        parser.add_argument('--context_backtrans_dir', type=str, default='QA/trans_{0}_en_context.txt'.format(lang))
    
    else: 
        parser = argparse.ArgumentParser()
        parser.add_argument('--queries_input_dir', type=str, default='QA/ood/sample_queries.txt')
        parser.add_argument('--queries_trans_dir', type=str, default='QA/ood/trans_en_{0}_queries.txt'.format(lang))
        parser.add_argument('--queries_backtrans_dir', type=str, default='QA/ood/trans_{0}_en_queries.txt'.format(lang))
    
        parser.add_argument('--context_input_dir', type=str, default='QA/ood/sample_context.txt')
        parser.add_argument('--context_trans_dir', type=str, default='QA/ood/trans_en_{0}_context.txt'.format(lang))
        parser.add_argument('--context_backtrans_dir', type=str, default='QA/ood/trans_{0}_en_context.txt'.format(lang))
    
    args = parser.parse_args()
    return args
