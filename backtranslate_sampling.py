from args import get_train_test_args
import argparse
from backtranslate_util import sample_dataset, concat_context, concat_queries
import util
from transformers import DistilBertTokenizerFast

# def prepare_eval_data(dataset_dict, tokenizer):
#     tokenized_examples = tokenizer(dataset_dict['question'],
#                                    dataset_dict['context'],
#                                    truncation="only_second",
#                                    stride=128,
#                                    max_length=384,
#                                    return_overflowing_tokens=True,
#                                    return_offsets_mapping=True,
#                                    padding='max_length')
#     # Since one example might give us several features if it has a long context, we need a map from a feature to
#     # its corresponding example. This key gives us just that.
#     sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

#     # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
#     # corresponding example_id and we will store the offset mappings.
#     tokenized_examples["id"] = []
#     for i in tqdm(range(len(tokenized_examples["input_ids"]))):
#         # Grab the sequence corresponding to that example (to know what is the context and what is the question).
#         sequence_ids = tokenized_examples.sequence_ids(i)
#         # One example can give several spans, this is the index of the example containing this span of text.
#         sample_index = sample_mapping[i]
#         tokenized_examples["id"].append(dataset_dict["id"][sample_index])
#         # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
#         # position is part of the context or not.
#         tokenized_examples["offset_mapping"][i] = [
#             (o if sequence_ids[k] == 1 else None)
#             for k, o in enumerate(tokenized_examples["offset_mapping"][i])
#         ]

#     return tokenized_examples



# def prepare_train_data(dataset_dict, tokenizer):
#     tokenized_examples = tokenizer(dataset_dict['question'],
#                                    dataset_dict['context'],
#                                    truncation="only_second",
#                                    stride=128,
#                                    max_length=384,
#                                    return_overflowing_tokens=True,
#                                    return_offsets_mapping=True,
#                                    padding='max_length')
#     sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
#     offset_mapping = tokenized_examples["offset_mapping"]

#     # Let's label those examples!
#     tokenized_examples["start_positions"] = []
#     tokenized_examples["end_positions"] = []
#     tokenized_examples['id'] = []
#     inaccurate = 0
#     for i, offsets in enumerate(tqdm(offset_mapping)):
#         # We will label impossible answers with the index of the CLS token.
#         input_ids = tokenized_examples["input_ids"][i]
#         cls_index = input_ids.index(tokenizer.cls_token_id)

#         # Grab the sequence corresponding to that example (to know what is the context and what is the question).
#         sequence_ids = tokenized_examples.sequence_ids(i)

#         # One example can give several spans, this is the index of the example containing this span of text.
#         sample_index = sample_mapping[i]
#         answer = dataset_dict['answer'][sample_index]
#         # Start/end character index of the answer in the text.
#         start_char = answer['answer_start'][0]
#         end_char = start_char + len(answer['text'][0])
#         tokenized_examples['id'].append(dataset_dict['id'][sample_index])
#         # Start token index of the current span in the text.
#         token_start_index = 0
#         while sequence_ids[token_start_index] != 1:
#             token_start_index += 1

#         # End token index of the current span in the text.
#         token_end_index = len(input_ids) - 1
#         while sequence_ids[token_end_index] != 1:
#             token_end_index -= 1

#         # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
#         if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
#             tokenized_examples["start_positions"].append(cls_index)
#             tokenized_examples["end_positions"].append(cls_index)
#         else:
#             # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
#             # Note: we could go after the last offset if the answer is the last word (edge case).
#             while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
#                 token_start_index += 1
#             tokenized_examples["start_positions"].append(token_start_index - 1)
#             while offsets[token_end_index][1] >= end_char:
#                 token_end_index -= 1
#             tokenized_examples["end_positions"].append(token_end_index + 1)
#             # assertion to check if this checks out
#             context = dataset_dict['context'][sample_index]
#             offset_st = offsets[tokenized_examples['start_positions'][-1]][0]
#             offset_en = offsets[tokenized_examples['end_positions'][-1]][1]
#             if context[offset_st : offset_en] != answer['text'][0]:
#                 inaccurate += 1

#     total = len(tokenized_examples['id'])
#     print(f"Preprocessing not completely accurate for {inaccurate}/{total} instances")
#     return tokenized_examples

# def read_and_process(args, tokenizer, dataset_dict, dir_name, dataset_name, split):
#     #TODO: cache this if possible
#     cache_path = f'{dir_name}/{dataset_name}_encodings.pt'
#     if os.path.exists(cache_path) and not args.recompute_features:
#         tokenized_examples = util.load_pickle(cache_path)
#     else:
#         if split=='train':
#             tokenized_examples = prepare_train_data(dataset_dict, tokenizer)
#         else:
#             tokenized_examples = prepare_eval_data(dataset_dict, tokenizer)
#         util.save_pickle(tokenized_examples, cache_path)
#     return tokenized_examples

#def get_sampling_dataset(args, datasets, data_dir, tokenizer, split_name):
    # for testing purpose can de-function the code and uncomment the line below
args = get_train_test_args() 
dataset_dict, sample_idx, sample_context_individual_length = sample_dataset(args, args.train_datasets, 
                                                                            args.train_dir,
                                                                            args.sample_prob, args.seed,
                                                                            args.sample_queries_dir,
                                                                            args.sample_context_dir)
print('Sampled queries are being saved at:', args.sample_queries_dir)         
print('Sampled context are being saved at:', args.sample_context_dir)                                                                      

backtranslated_queries = concat_queries(args.backtranslate_queries_dir)
backtranslated_context = concat_context(args.backtranslate_context_dir, sample_context_individual_length)

new_dataset_dict = dict(dataset_dict)

for (index, replacement) in zip(sample_idx, backtranslated_queries):
    new_dataset_dict['question'][index] = replacement

for (index, replacement) in zip(sample_idx, backtranslated_context):
    new_dataset_dict['context'][index] = replacement

    # for testing purpose can comment out the two lines below and check new_dataset_dict
    # data_encodings = read_and_process(args, tokenizer, new_dataset_dict, data_dir, dataset_name, split_name)
    # return util.QADataset(data_encodings, train=(split_name=='train')), dataset_dict

# if __name__ == '__main__':
#     args = get_train_test_args()
#     tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
#     output = get_sampling_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train')






    



