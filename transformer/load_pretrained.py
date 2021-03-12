import torch
from args import get_queries_args, get_context_args
from tqdm import tqdm
# pip install Cython
# pip install hydra-core
# pip install sacremoses

def en_de(input_dir, output_dir):
    en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de',
                       checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                       tokenizer='moses', bpe='fastbpe')
    en2de.eval()
    en2de.cuda()
    run(en2de, args.input_dir, args.trans_dir)

    
def de_en(args):
    de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en',
                       checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                       tokenizer='moses', bpe='fastbpe')
    de2en.eval()
    de2en.cuda()
    run(de2en, args.trans_dir, args.backtrans_dir)
    

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
    print("========= Translating queries =========")
    queries_args = get_queries_args()
    en_de(queries_args.input_dir, queries_args.trans_dir)
    de_en(queries_args.trans_dir, queries_args.backtrans_dir)
    
#     print("========= Translating context =========")
#     context_args = get_context_args()
#     backtranslate(context_args)
#     en_de(context_args.input_dir, context_args.trans_dir)
#     de_en(context_args.trans_dir, context_args.backtrans_dir)

# Test
# de_trans = en2de.translate('Hello world!')
# print("EN-DE:", de_trans)
# back_trans = de2en.translate(de_trans)
# print("DE-EN:", back_trans)
