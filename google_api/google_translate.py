import torch
from args import get_queries_args, get_context_args
from tqdm import tqdm
from googletrans import Translator
# pip install googletrans==3.1.0a0

def en_es(input_dir, output_dir):
    translator_en2es = Translator(src = "en", dest = "es")
    run(translator_en2es, input_dir, output_dir)

    
def es_en(input_dir, output_dir):
    translator_es2en = Translator(src = "es", dest = "en")
    run(translator_es2en, input_dir, output_dir)

def run(model, input_dir, output_dir):
    in_file = open(input_dir, 'r')
    out_file = open(output_dir, 'w')
    
    for line in tqdm(in_file, desc='Translating'):
        trans = model.translate(line).text
        out_file.write(trans + "\n")

    in_file.close()
    out_file.close()
    print("Translations stored at:", output_dir)


if __name__ == '__main__':
    print("========= Translating queries =========")
    queries_args = get_queries_args(lang='es')
    en_es(queries_args.input_dir, queries_args.trans_dir)
    es_en(queries_args.trans_dir, queries_args.backtrans_dir)
