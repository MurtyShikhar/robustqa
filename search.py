import os
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from datetime import datetime

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
    args["adam_weight_decay"] = tune.loguniform(1e-4, 1e-1)
    args["adv_loss_weight"] = tune.loguniform(5e-3, 5e-1)

    # discriminator parameters
    args["discriminator_lr"] = tune.loguniform(3e-6, 3e-4)
    args["discriminator_momentum"] = tune.uniform(0.8, 0.95)
    args["discriminator_step_multiplier"] = tune.randint(1, 4)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    scheduler = ASHAScheduler(
        metric="oodomain_F1",
        mode="max",
        max_t=args["max_num_batches"],
        grace_period=args["min_num_batches"],
        reduction_factor=2 # taken from pytorch tutorial
    )

    timestamp = datetime.now().strftime("%m_%d_%Y_%I_%M_%S_%p")
    result = tune.run(
        partial(do_train, tokenizer=tokenizer),
        config=args,
        name=f'{args["tune_name"]}_{timestamp}',
        local_dir="tune_results",
        num_samples=args["num_tune_samples"],
        resources_per_trial={"cpu": args["num_cpu_per_test"], "gpu": args["num_gpu_per_test"]},
        scheduler=scheduler
    )

if __name__ == "__main__":
    main()
