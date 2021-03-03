import args
import json

from collections import Counter

def read_data(directory, datasets):
    dataset_dict = {}
    for dataset in datasets:
        curr_dataset = json.loads(open(f'{directory}/{dataset}').read())
        dataset_dict[dataset] = curr_dataset

    topics_list = []
    for dataset in datasets:
        for entry in dataset_dict[dataset]["data"]:
            if "topic" in entry:
                topics_list.append(entry["topic"])
            else:
                topics_list.append(dataset)
    
    return dataset_dict, topics_list

def create_subsets(directory, datasets, keep_percentage=1):
    print(f'Creating subsets in {directory}')
    dataset_dict, topics_list = read_data(directory, datasets)
    
    # compute count of entries per topic
    # multiply by subset keep percentage to calculate num of entries to keep
    topics_count = dict(Counter(topics_list))
    for key in topics_count.keys():
        # keep at least 3 per topic
        topics_count[key] = max(3, int(topics_count[key] * keep_percentage))
    
    for dataset in datasets:
        dataset_subset = []
        print(f'{dataset} original size: {len(dataset_dict[dataset]["data"])}')

        for entry in dataset_dict[dataset]["data"]:
            topic = dataset
            if "topic" in entry:
                topic = entry["topic"]

            if topics_count[topic] > 0:
                topics_count[topic] -= 1
                dataset_subset.append(entry)
            
        print(f'{dataset} subset size: {len(dataset_subset)}')

        subset_dict = {"data": dataset_subset}
        with open(f'{directory}/{dataset}_subset', 'w') as f:
            json.dump(subset_dict, f)

    print()

if __name__ == "__main__":
    args = args.get_train_test_args()
    datasets = args["train_datasets"].split(',')

    create_subsets(args["train_dir"], datasets, args["subset_keep_percentage"])
    create_subsets(args["val_dir"], datasets)
