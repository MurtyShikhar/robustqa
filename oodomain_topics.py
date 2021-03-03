import json

if __name__ == "__main__":
    datasets = ["duorc", "race", "relation_extraction"]
    for dataset in datasets:
        path = f'datasets/oodomain_train/{dataset}'
        with open(path, 'rb') as f:
            squad_dict = json.load(f)

        for group in squad_dict['data']:
            group['topic'] = dataset

        path = f'datasets/oodomain_train/{dataset}_plus_topics'
        with open(path, 'w') as f:
            json.dump(squad_dict, f)