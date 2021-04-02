import torch
from google_args import get_google_args
from tqdm import tqdm
from googletrans import Translator
import time
# pip install googletrans==3.1.0a0

def en_es(input_dir, output_dir):
    translator_en2es = Translator()
    run(translator_en2es, input_dir, output_dir, src = "en", dest = "es")

    
def es_en(input_dir, output_dir):
    translator_es2en = Translator()
    run(translator_es2en, input_dir, output_dir, src = "es", dest = "en")

def run(input_dir, output_dir, src, dest):
    in_file = open(input_dir, 'r')
    out_file = open(output_dir, 'w')
    model = Translator(service_urls=['translate.googleapis.com'])

    i = 0
    for line in tqdm(in_file.readlines(), desc='Translating'):
        if i % 100 == 0:
            time.sleep(0.5)
            model = Translator(service_urls=['translate.googleapis.com'])
        trans = model.translate(line, src = src, dest = dest).text
        i += 1
        out_file.write(trans + "\n")

    in_file.close()
    out_file.close()
    print("Translations stored at:", output_dir)


if __name__ == '__main__':
    args = get_google_args(lang='es')
    print("========= Translating queries (es) =========")
    run(args.queries_input_dir, args.queries_trans_dir, src = "en", dest = "es")
    run(args.queries_trans_dir, args.queries_backtrans_dir, src = "en", dest = "es")
    # print("========= Translating context (es) =========")
    # en_es(args.context_input_dir, args.context_trans_dir)
    # es_en(args.context_trans_dir, args.context_backtrans_dir)
