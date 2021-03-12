import argparse

def get_queries_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='transformer/')
    parser.add_argument('--trans_dir', type=str, default='transformer/')
    parser.add_argument('--backtrans_dir', type=str, default='transformer/')
    args = parser.parse_args()
    return args

def get_context_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='transformer/')
    parser.add_argument('--trans_dir', type=str, default='transformer/')
    parser.add_argument('--backtrans_dir', type=str, default='transformer/')
    args = parser.parse_args()
    return args