import argparse

def get_queries_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='../2_layer_nmt/QA/sample_queries.txt')
    parser.add_argument('--trans_dir', type=str, default='transformer/trans_en_de_queries.txt')
    parser.add_argument('--backtrans_dir', type=str, default='transformer/trans_de_en_queries.txt')
    args = parser.parse_args()
    return args

def get_context_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='../2_layer_nmt/QA/sample_context.txt')
    parser.add_argument('--trans_dir', type=str, default='transformer/trans_en_de_context.txt')
    parser.add_argument('--backtrans_dir', type=str, default='transformer/trans_de_en_context.txt')
    args = parser.parse_args()
    return args

