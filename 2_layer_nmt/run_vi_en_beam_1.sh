#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run_beam.py train --src-lang=vi --tgt-lang=en --vocab=vocab.json --cuda --lr=7.5e-4 --patience=2 --valid-niter=200 --batch-size=64 --dropout=.2 --hidden-size=512 --embed-size=512
elif [ "$1" = "test" ]; then
        CUDA_VISIBLE_DEVICES=0 python run_beam.py decode --src-lang=vi --tgt-lang=en --beam-size=1 model.bin test_outputs_beam_1.txt --cuda
elif [ "$1" = "train_local" ]; then
	python run_beam.py train --src-lang=vi --tgt-lang=en --vocab=vocab.json --lr=7.5e-4 --patience=2 --valid-niter=200 --batch-size=64 --dropout=.2 --hidden-size=512 --embed-size=512
elif [ "$1" = "translate_queries" ]; then
        CUDA_VISIBLE_DEVICES=0 python run_beam.py translate --src-lang=vi --tgt-lang=en --beam-size=1 ./QA/trans_en_vi_queries_beam_1_dropped.txt model.bin ./QA/trans_vi_en_queries_beam_1.txt --cuda
elif [ "$1" = "translate_context" ]; then
        CUDA_VISIBLE_DEVICES=0 python run_beam.py translate --src-lang=vi --tgt-lang=en --beam-size=1 ./QA/trans_en_vi_context_beam_1_dropped.txt model.bin ./QA/trans_vi_en_context_beam_1.txt --cuda
elif [ "$1" = "vocab" ]; then
	python vocab.py --src-lang=vi --tgt-lang=en --src-train-vocab-size=8000 --tgt-train-vocab-size=16000
else
	echo "Invalid Option Selected"
fi

