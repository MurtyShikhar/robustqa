import argparse

def get_train_test_args():
    parser = argparse.ArgumentParser()
    # hyperparameters
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--adam-weight-decay', type=float, default=0)
    parser.add_argument('--discriminator-lr', type=float, default=3e-5)
    parser.add_argument('--discriminator-momentum', type=float, default=0.8)
    parser.add_argument('--adv-loss-weight', type=float, default=0.01)
    parser.add_argument('--discriminator-step-multiplier', type=int, default=1)

    parser.add_argument('--num-visuals', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='save/')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--train-datasets', type=str, default='squad,nat_questions,newsqa')
    parser.add_argument('--oodomain-train-datasets', type=str, default='duorc,race,relation_extraction')
    parser.add_argument('--run-name', type=str, default='multitask_distilbert')
    parser.add_argument('--recompute-features', action='store_true')
    parser.add_argument('--train-dir', type=str, default='datasets/indomain_train')
    parser.add_argument('--oodomain-train-dir', type=str, default='datasets/oodomain_train')
    parser.add_argument('--val-dir', type=str, default='datasets/indomain_val')
    parser.add_argument('--oodomain-val-dir', type=str, default='datasets/oodomain_val')
    parser.add_argument('--eval-dir', type=str, default='datasets/oodomain_test')
    parser.add_argument('--eval-datasets', type=str, default='race,relation_extraction,duorc')
    parser.add_argument('--do-train', action='store_true')
    parser.add_argument('--do-eval', action='store_true')
    parser.add_argument('--sub-file', type=str, default='')
    parser.add_argument('--visualize-predictions', action='store_true')
    parser.add_argument('--eval-every', type=int, default=5000)
    parser.add_argument('--subset-keep-percentage', type=float, default=0.01)

    # arguments for hyperparameter search
    parser.add_argument("--tune-name", type=str, default="hyperparam-search")
    parser.add_argument('--num-gpu-per-test', type=int, default=0)
    parser.add_argument('--num-cpu-per-test', type=int, default=1)
    parser.add_argument('--num-tune-samples', type=int, default=10)
    parser.add_argument('--tune-checkpoint-path', type=str)

    parser.add_argument('--tune', action='store_true')

    args = parser.parse_args()
    return vars(args) # return as dict to support hyperparam search
