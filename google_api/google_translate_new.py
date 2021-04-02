import torch
from google_args import get_google_args
from tqdm import tqdm
from google_trans_new import google_translator
import time
# pip install google_trans_new

def run(input_dir, output_dir, src, dest):
    in_file = open(input_dir, 'r')
    out_file = open(output_dir, 'w')
    model = google_translator()

    i = 0
    for line in tqdm(in_file.readlines(), desc='Translating'):
        trans = model.translate(line, lang_src = src, lang_tgt = dest)
        time.sleep(0.1)
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
