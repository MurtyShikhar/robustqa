#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --src-lang=en --tgt-lang=de --vocab=vocab.json --cuda --lr=1e-3 --patience=5 --valid-niter=200 --batch-size=32 --dropout=.2 --hidden-size=512 --embed-size=512
elif [ "$1" = "test" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py decode --src-lang=en --tgt-lang=de model.bin test_outputs.txt --cuda
elif [ "$1" = "train_local" ]; then
	python run.py train --src-lang=en --tgt-lang=de --vocab=vocab.json --lr=1e-3 --patience=5 --valid-niter=200 --batch-size=32 --dropout=.2 --hidden-size=512 --embed-size=512
elif [ "$1" = "translate_queries" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py translate --src-lang=en --tgt-lang=de ./QA/sample_queries.txt model.bin ./QA/trans_en_de_queries.txt --cuda
elif [ "$1" = "translate_context" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py translate --src-lang=en --tgt-lang=de ./QA/sample_context.txt model.bin ./QA/trans_en_de_context.txt --cuda
elif [ "$1" = "vocab" ]; then
	python vocab.py --src-lang=en --tgt-lang=de --src-train-vocab-size=12000 --tgt-train-vocab-size=12000
else
	echo "Invalid Option Selected"
fi

