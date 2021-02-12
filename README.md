## Starter code for robustqa track
- Download datasets from [here](https://drive.google.com/file/d/15jBQ4d62NpYfwoDBw39AY6YcTnacO6Df/view?usp=sharing)
- Setup environment with `conda env create -f environment.yml`
- Train a baseline MTL system with `python train.py --do-train --eval-every 2000 --run-name baseline`
- Evaluate the system on test set with `python train.py --do-eval --sub-file mtl_submission.csv --save-dir baseline-01`
- Share the csv file in `save/baseline-01` folder with smurty
