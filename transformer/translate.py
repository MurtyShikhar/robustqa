import torch
from args import get_transformer_args
from tqdm import tqdm
import sys
# pip install Cython
# pip install hydra-core
# pip install sacremoses

def transformer_19(src, tgt, input_dir, output_dir):
    model = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.{0}-{1}'.format(src, tgt),
                       checkpoint_file='model1.pt', tokenizer='moses', bpe='fastbpe')
    model.eval()
    model.cuda()
    run(model, input_dir, output_dir)


def pretrained(type, year, src, tgt, input_dir, output_dir):
    if year == 19:
        model = torch.hub.load('pytorch/fairseq', '{0}.wmt{1}.{2}-{3}'.format(type, year, src, tgt),
                       checkpoint_file='model1.pt', tokenizer='moses', bpe='fastbpe')
    else:
        model = torch.hub.load('pytorch/fairseq', '{0}.wmt{1}.{2}-{3}'.format(type, year, src, tgt),
                       tokenizer='moses', bpe='subword_nmt')
    model.eval()
    model.cuda()
    run(model, input_dir, output_dir)

    
def run(model, input_dir, output_dir):
    in_file = open(input_dir, 'r')
    out_file = open(output_dir, 'w')

    for line in tqdm(in_file, desc='Translating', file=sys.stdout):
        trans = model.translate(line)
        out_file.write(trans + "\n")

    in_file.close()
    out_file.close()
    print("Translations stored at:", output_dir)


if __name__ == '__main__':
    args = get_transformer_args(lang='de')
    print("========= Translating queries (de) =========")
    transformer_19('en', 'de', args.queries_input_dir, args.queries_trans_dir)
    transformer_19('de', 'en', args.queries_trans_dir, args.queries_backtrans_dir)
#     print("========= Translating context (de) =========")
#     transformer_19('en', 'de', args.context_input_dir, args.context_trans_dir)
#     transformer_19('de', 'en', args.context_trans_dir, args.context_backtrans_dir)

    print("========= Translating queries (ru) =========")
    args = get_transformer_args(lang='ru')
    transformer_19('en', 'ru', args.queries_input_dir, args.queries_trans_dir)
    transformer_19('ru', 'en', args.queries_trans_dir, args.queries_backtrans_dir) 
#     print("========= Translating context (ru) =========")
#     transformer_19('en', 'ru', args.context_input_dir, args.context_trans_dir)
#     transformer_19('ru', 'en', args.context_trans_dir, args.context_backtrans_dir)
