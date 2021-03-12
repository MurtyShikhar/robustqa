import torch
from args import get_queries_args, get_context_args
from tqdm import tqdm
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

    for line in tqdm(in_file, desc='Translating'):
        trans = model.translate(line)
        out_file.write(trans + "\n")

    in_file.close()
    out_file.close()
    print("Translations stored at:", output_dir)


if __name__ == '__main__':
    print("========= Translating queries (de) =========")
    queries_args = get_queries_args(lang='de')
    transformer_19('en', 'de', queries_args.input_dir, queries_args.trans_dir)
    transformer_19('de', 'en', queries_args.trans_dir, queries_args.backtrans_dir)
    
#     print("========= Translating context (de) =========")
#     context_args = get_context_args(lang='de')
#     transformer_19('en', 'de', context_args.input_dir, context_args.trans_dir)
#     transformer_19('de', 'en', context_args.trans_dir, context_args.backtrans_dir)

#     print("========= Translating queries (ru) =========")
#     queries_args = get_queries_args(lang='ru')
#     transformer_19('en', 'ru', queries_args.input_dir, queries_args.trans_dir)
#     transformer_19('ru', 'en', queries_args.trans_dir, queries_args.backtrans_dir)
    
#     print("========= Translating context (ru) =========")
#     context_args = get_context_args(lang='ru')
#     transformer_19('en', 'ru', context_args.input_dir, context_args.trans_dir)
#     transformer_19('ru', 'en', context_args.trans_dir, context_args.backtrans_dir)
