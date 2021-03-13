import torch
from transformer_args import get_transformer_args
from tqdm import tqdm
# pip install Cython
# pip install hydra-core
# pip install sacremoses
# pip install fastBPE

def en_de(input_dir, output_dir):
    en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de',
                       checkpoint_file='model1.pt', tokenizer='moses', bpe='fastbpe')
    en2de.eval()
    en2de.cuda()
    run(en2de, input_dir, output_dir, "en to de")

    
def de_en(input_dir, output_dir):
    de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en',
                       checkpoint_file='model1.pt', tokenizer='moses', bpe='fastbpe')
    de2en.eval()
    de2en.cuda()
    run(de2en, input_dir, output_dir, "de to en")
    

def en_ru(input_dir, output_dir):
    en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru',
                       checkpoint_file='model1.pt', tokenizer='moses', bpe='fastbpe')
    en2ru.eval()
    en2ru.cuda()
    run(en2ru, input_dir, output_dir, "en to ru")
    
    
def ru_en(input_dir, output_dir):
    ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en',
                       checkpoint_file='model1', tokenizer='moses', bpe='fastbpe')
    ru2en.eval()
    ru2en.cuda()
    run(ru2en, input_dir, output_dir, "ru to en")
    
    
def run(model, input_dir, output_dir, src_tgt):
    in_file = open(input_dir, 'r').readlines()
    out_file = open(output_dir, 'w')
   
    with tqdm(len(in_file), desc='Translating: ' + src_tgt):
        for line in in_file:
            trans = model.translate(line)
            out_file.write(trans + "\n")

    in_file.close()
    out_file.close()
    print("Translations stored at:", output_dir)


if __name__ == '__main__':
    args = get_transformer_args(lang='de')
#    print("========= Translating queries (de) =========")
#    en_de(args.queries_input_dir, args.queries_trans_dir)
#    de_en(args.queries_trans_dir, args.queries_backtrans_dir)
    print("========= Translating context (de) =========")
#     en_de(args.context_input_dir, args.context_trans_dir)
    de_en(args.context_trans_dir, args.context_backtrans_dir)

#     args = get_transformer_args(lang='ru')
#     print("========= Translating queries (ru) =========")
#     en_ru(args.queries_input_dir, args.queries_trans_dir)
#     ru_en(args.queries_trans_dir, args.queries_backtrans_dir) 
#     print("========= Translating context (ru) =========")
#     en_ru(args.context_input_dir, args.context_trans_dir)
#     ru_en(args.context_trans_dir, args.context_backtrans_dir)


# Test
# de_trans = en2de.translate('Hello world!')
# print("EN-DE:", de_trans)
# back_trans = de2en.translate(de_trans)
# print("DE-EN:", back_trans)
