import os
from ray import tune
from functools import partial

from transformers import DistilBertTokenizerFast

from args import get_train_test_args
from train import do_train

def main():
    if not os.path.exists("tune_results"):
        os.makedirs("tune_results")

    args = get_train_test_args()
    args["tune"] = True

    # qa model parameters
    args["lr"] = tune.loguniform(3e-6, 3e-4)
    args["batch_size"] = tune.choice(args["tune_batch_sizes"])
    args["seed"] = tune.randint(1, 100)
    args["adam_weight_decay"] = 0
    args["adv_loss_weight"] = tune.loguniform(5e-3, 5e-1)

    # discriminator parameters
    args["discriminator_lr"] = tune.loguniform(3e-6, 3e-4)
    args["discriminator_momentum"] = tune.uniform(0.8, 0.95)
    args["discriminator_step_multiplier"] = tune.randint(1, 4)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    result = tune.run(
        partial(do_train, tokenizer=tokenizer),
        config=args,
        local_dir="tune_results",
        num_samples=args["num_tune_samples"],
        resources_per_trial={"cpu": args["num_cpu_per_test"], "gpu": args["num_gpu_per_test"]},
    )
    
    # trial results automatically get logged by tune
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {0}".format(best_trial.config))
    print("Best trial final loss: {0}".format(best_trial.last_result))

if __name__ == "__main__":
    main()
