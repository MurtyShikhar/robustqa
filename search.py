import os
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from datetime import datetime

from transformers import DistilBertTokenizerFast

from args import get_train_test_args
from train import do_train

from search_analysis import plot_results

def do_search():
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
    results = tune.run(
        partial(do_train, tokenizer=tokenizer),
        config=args,
        name=f'{args["tune_name"]}_{timestamp}',
        local_dir="tune_results",
        num_samples=args["num_tune_samples"],
        resources_per_trial={"cpu": args["num_cpu_per_test"], "gpu": args["num_gpu_per_test"]},
        scheduler=scheduler
    )

    return results, f'{args["tune_name"]}_{timestamp}'

if __name__ == "__main__":
    results, name = do_search()

    trials = list(results.trial_dataframes.keys())
    results_dict = dict()
    for trial in trials:
        trial_id = results.trial_dataframes[trial]["trial_id"][0]
        results_dict[trial_id] = results.trial_dataframes[trial]
    results_dict[list(results_dict.keys())[0]].columns

    plot_results(results_dict, f'tune_results/{name}')
