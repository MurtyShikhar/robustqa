from ray.tune import Analysis
import matplotlib.pyplot as plt
import sys

def plot_metric(axis, dataframe, trial_id, metric_label):
    x = list(dataframe.index)
    metric = dataframe[metric_label]

    line = axis.plot(x, metric, label=trial_id)
    axis.set_xticks(x)
    axis.set_title(metric_label, fontsize=14)
    
    return line

def plot_results(results_dict, root_dir):
    metrics = ["loss", "indomain_EM", "indomain_QALoss"
        , "discriminator_accuracy", "indomain_F1", "oodomain_QALoss"
        , "discriminator_loss", "oodomain_EM", "indomain_KLDiv"
        , "discriminator_kl_div", "oodomain_F1", "oodomain_KLDiv"]
    fig, (row_one, row_two, row_three, row_four)= plt.subplots(4, 3, figsize=(24, 16))
    axes = [row_one[0], row_one[1], row_one[2]
        , row_two[0], row_two[1], row_two[2]
        , row_three[0], row_three[1], row_three[2]
        , row_four[0], row_four[1], row_four[2]]

    lines = {}
    for trial_id in results_dict:
        for i, metric in enumerate(metrics):
            line = plot_metric(axes[i], results_dict[trial_id], trial_id, metric)
            lines[trial_id] = line

    fig.legend(list(lines.values()), labels=list(lines.keys()), loc="center right", fontsize=14, ncol=1)
    fig.suptitle("Hyperparameter Search Results", x=0.15, y=.95, fontsize=24)
    fig.subplots_adjust(top=.9, left=0.05, right=0.875)

    fig.savefig(f'{root_dir}/results_plot.png')

if __name__ == "__main__":
    results_dir = sys.argv[1] # ""
    results = Analysis(results_dir)
    trials = list(results.trial_dataframes.keys())
    results_dict = dict()
    for trial in trials:
        trial_id = results.trial_dataframes[trial]["trial_id"][0]
        results_dict[trial_id] = results.trial_dataframes[trial]
    results_dict[list(results_dict.keys())[0]].columns

    plot_results(results_dict, results_dir)