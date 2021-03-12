import torch
from args import get_queries_args, get_context_args
from tqdm import tqdm
# pip install Cython
# pip install hydra-core
# pip install sacremoses

def en_de(input_dir, output_dir):
    en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de',
                       checkpoint_file='model1.pt', tokenizer='moses', bpe='fastbpe')
    en2de.eval()
    en2de.cuda()
    run(en2de, input_dir, output_dir)

    
def de_en(input_dir, output_dir):
    de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en',
                       checkpoint_file='model1.pt', tokenizer='moses', bpe='fastbpe')
    de2en.eval()
    de2en.cuda()
    run(de2en, input_dir, output_dir)
    

def en_ru(input_dir, output_dir):
    en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru',
                       checkpoint_file='model1.pt', tokenizer='moses', bpe='fastbpe')
    en2ru.eval()
    en2ru.cuda()
    run(en2ru, input_dir, output_dir)
    
    
def ru_en(input_dir, output_dir):
    ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en',
                       checkpoint_file='model1', tokenizer='moses', bpe='fastbpe')
    ru2en.eval()
    ru2en.cuda()
    run(ru2en, input_dir, output_dir)
    
    
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
    en_de(queries_args.input_dir, queries_args.trans_dir)
    de_en(queries_args.trans_dir, queries_args.backtrans_dir)
    
#     print("========= Translating context (de) =========")
#     context_args = get_context_args(lang='de')
#     en_de(context_args.input_dir, context_args.trans_dir)
#     de_en(context_args.trans_dir, context_args.backtrans_dir)

#     print("========= Translating queries (ru) =========")
#     queries_args = get_queries_args(lang='ru')
#     en_ru(queries_args.input_dir, queries_args.trans_dir)
#     ru_en(queries_args.trans_dir, queries_args.backtrans_dir)
    
#     print("========= Translating context (ru) =========")
#     context_args = get_context_args(lang='ru')
#     en_ru(context_args.input_dir, context_args.trans_dir)
#     ru_en(context_args.trans_dir, context_args.backtrans_dir)

# Test
# de_trans = en2de.translate('Hello world!')
# print("EN-DE:", de_trans)
# back_trans = de2en.translate(de_trans)
# print("DE-EN:", back_trans)
