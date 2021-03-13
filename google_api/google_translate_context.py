import torch
from args import get_queries_args, get_context_args
from tqdm import tqdm
from googletrans import Translator
# pip install googletrans==3.1.0a0

def en_es(input_dir, output_dir):
    translator_en2es = Translator()
    run(translator_en2es, input_dir, output_dir, src = "en", dest = "es")

    
def es_en(input_dir, output_dir):
    translator_es2en = Translator()
    run(translator_es2en, input_dir, output_dir, src = "es", dest = "en")

def run(model, input_dir, output_dir, src, dest):
    in_file = open(input_dir, 'r')
    out_file = open(output_dir, 'w')
    
    for line in tqdm(in_file.readlines(), desc='Translating'):
        trans = model.translate(line, src = src, dest = dest).text
        out_file.write(trans + "\n")

    in_file.close()
    out_file.close()
    print("Translations stored at:", output_dir)


if __name__ == '__main__':
    print("========= Translating context =========")
    context_args = get_context_args(lang='es')
    en_es(context_args.input_dir, context_args.trans_dir)
    es_en(context_args.trans_dir, context_args.backtrans_dir)
