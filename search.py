import os
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from datetime import datetime

from transformers import DistilBertTokenizerFast

from args import get_train_test_args
from train import do_train

def plot_results(dataframe, params, colors):
    x = list(dataframe.index)
    f1 = dataframe["F1"]
    trial_id = dataframe["trial_id"][0]

    plt.plot(x, f1, label=trial_id, color=colors[0])
    plt.title("F1 Scores")
    plt.ylabel("F1")
    plt.legend(loc="upper right") 

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
        metric="F1",
        mode="max",
        max_t=100, # number of eval_every batches
        grace_period=10,
        reduction_factor=2
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
    
    trials = list(results.trial_dataframes.keys())
    results_dict = dict()
    for trial in trials:
        params = json.loads(open(f'{trial}/params.json').read())
        results = pandas.read_json(f'{trial}/result.json', orient='records', lines=True)
        results_dict[trial] = (params, results)

    plt.figure(figsize=(15, 12))
    for trial in results_dict:
        plot_results(results_dict[trial][2], results_dict[trial][0])
    plt.savefig(f'tune_results/{args["tune_name"]}_{timestamp}/search_f1_scores.png')

if __name__ == "__main__":
    main()
