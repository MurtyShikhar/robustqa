## Starter code for robustqa track
- Download datasets from [here](https://drive.google.com/file/d/1IcRXgflri-dwziR-BiQIu55PaKX9Q_ys/view?usp=sharing) or an even smaller version from [here](https://drive.google.com/file/d/15jBQ4d62NpYfwoDBw39AY6YcTnacO6Df/view?usp=sharing)
- Setup environment with `conda env create -f environment.yml`
- Train a baseline MTL system with `python train.py --do_train --eval-every 2000`
- Evaluate the system on test set with `python train.py --do_test --sub_file mtl_submission.csv`
- Share the csv file in `save/test` folder with smurty
