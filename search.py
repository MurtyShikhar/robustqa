import numpy as np
from ray import tune

from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering

import util
from args import get_train_test_args
from train import do_train

# TODO: save hyperparameters that you've tried

def main():
    args = vars(get_train_test_args())

    args["lr"] = tune.loguniform(1e-4, 1e-1)
    args["batch_size"] = tune.choice([2, 4, 8, 16, 32, 64, 128, 256])
    args["seed"] = tune.sample_from(lambda _: 1 * np.random.randint(1, 100))
    args["adamw"]

    print(args)
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    result = tune.run(
        do_train,
        args=args,
        model=model,
        tokenizer=tokenizer
    )

    print("Best config: ", result.get_best_config(
        metric="loss", mode="min")
    )

if __name__ == "__main__":
    main()