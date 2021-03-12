# def get_trans_context_answers(context_dir, sample_context_individual_length,
#                               gold_answers, answer_locs, output_context_dir):
#     """
#         input:
#             context_dir <file>: the back translated context file
#             sample_context_individual_length list<integer>: number of sentences in each context
#             gold_answers list<list<string>>: the list of gold answers
#             answer_locs list<list<integer>>: the list of answer locs
#             output_context_dir <file>: concatenated context
#         returns:
#             new_answers: list of new_answers
#     """
#     in_file = open(context_dir, 'r')
#     out_file = open(output_context_dir, 'w')

#     num_samples = len(sample_context_individual_length)
#     new_answers = []

#     for i in range(num_samples):
#         curr_answers = gold_answers[i]['text']
#         curr_locs = answer_locs[i]

#         new_start_idx = []
#         new_curr_answers = []
#         curr_context = ''
#         char_count = 0

#         for j in range(sample_context_individual_length[i]):
#             context_sent = in_file.readline().strip()

#             for k in range(len(curr_locs)):
#                 if j == curr_locs[k]:
#                     start_pos, best_substring = compute_answer_span(context_sent, curr_answers[k])
#                     new_start_idx.append(char_count + start_pos)
#                     new_curr_answers.append(best_substring)
            
#             curr_context += context_sent + " "
#             char_count += len(context_sent + " ")

#         new_answers.append(dict({'answer_start': new_start_idx, 'text': new_curr_answers}))
#         out_file.write(curr_context + '\n')
    
#     in_file.close()
#     out_file.close()
#     return new_answers


# def concat_queries(queries_dir):
#     output_queries = []
#     f = open(queries_dir, 'r')
#     whole_queries = f.readlines()
#     for q in whole_queries:
#         output_queries.append(q)
#     return output_queries

# def compute_backtrans_bleu(preds, refs):
#   bleu = sacrebleu.corpus_bleu(preds, [refs])
#   return bleu.score

        
# def compute_backtrans_bleu(original_file, backtrans_file):
#   ref_file = open(original_file, 'r')
#   pred_file = open(backtrans_file, 'r')
  
#   refs = [line.strip() for line in ref_file]
#   preds = [line.strip() for line in pred_file]
#   bleu = sacrebleu.corpus_bleu(preds, [refs])
#   return bleu.score


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
