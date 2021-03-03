#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --src-lang=en --tgt-lang=es --vocab=vocab.json --cuda --lr=7.5e-4 --patience=2 --valid-niter=200 --batch-size=32 --dropout=.3 --hidden-size=512 --embed-size=512
elif [ "$1" = "test" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py decode --src-lang=en --tgt-lang=es model.bin test_outputs.txt --cuda
elif [ "$1" = "train_local" ]; then
	python run.py train --src-lang=en --tgt-lang=es --vocab=vocab.json --lr=7.5e-4 --patience=2 --valid-niter=200 --batch-size=32 --dropout=.3 --hidden-size=512 --embed-size=512
elif [ "$1" = "translate_queries" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py translate --src-lang=en --tgt-lang=es ./QA/sample_queries.txt model.bin ./QA/trans_en_es_queries.txt --cuda
elif [ "$1" = "translate_context" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py translate --src-lang=en --tgt-lang=es ./QA/sample_context.txt model.bin ./QA/trans_en_es_context.txt --cuda
elif [ "$1" = "vocab" ]; then
	python vocab.py --src-lang=en --tgt-lang=es --src-train-vocab-size=8000 --tgt-train-vocab-size=13000
else
	echo "Invalid Option Selected"
fi

