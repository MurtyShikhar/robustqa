import argparse

def get_queries_args(lang='es'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='../2_layer_nmt/QA/sample_queries.txt')
    parser.add_argument('--trans_dir', type=str, default='trans_en_{0}_queries.txt'.format(lang))
    parser.add_argument('--backtrans_dir', type=str, default='trans_{0}_en_queries.txt'.format(lang))
    args = parser.parse_args()
    return args

def get_context_args(lang='es'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='../2_layer_nmt/QA/sample_context.txt')
    parser.add_argument('--trans_dir', type=str, default='trans_en_{0}_context.txt'.format(lang))
    parser.add_argument('--backtrans_dir', type=str, default='trans_{0}_en_context.txt'.format(lang))
    args = parser.parse_args()
    return args

