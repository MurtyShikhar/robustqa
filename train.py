import argparse
import json
import os
from collections import OrderedDict
from typing import Optional
import numpy as np

import torch
import csv
import util
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from args import get_train_test_args
from DomainAdversarial import DomainDiscriminator

from tqdm import tqdm

def prepare_test_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["id"] = []
    for i in tqdm(range(len(tokenized_examples["input_ids"]))):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["id"].append(dataset_dict["id"][sample_index])
        #tokenized_examples['topic_id'].append(dataset_dict['topic_id'][sample_index])
        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]
    return tokenized_examples

def prepare_eval_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples["offset_mapping"]

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples["id"] = []
    inaccurate = 0
    for i, offsets in enumerate(tqdm(offset_mapping)):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answer = dataset_dict['answer'][sample_index]
        # Start/end character index of the answer in the text.
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        tokenized_examples['id'].append(dataset_dict['id'][sample_index])
        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)
            # assertion to check if this checks out
            context = dataset_dict['context'][sample_index]
            offset_st = offsets[tokenized_examples['start_positions'][-1]][0]
            offset_en = offsets[tokenized_examples['end_positions'][-1]][1]
            if context[offset_st : offset_en] != answer['text'][0]:
                inaccurate += 1

    total = len(tokenized_examples['id'])
    print(f"Preprocessing not completely accurate for {inaccurate}/{total} instances")
    #tokenized_examples['topic_id'] = []
    '''
    for i in tqdm(range(len(tokenized_examples["input_ids"]))):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["id"].append(dataset_dict["id"][sample_index])
        #tokenized_examples['topic_id'].append(dataset_dict['topic_id'][sample_index])
        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]
    '''

    return tokenized_examples

def prepare_train_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    offset_mapping = tokenized_examples["offset_mapping"]

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples['id'] = []
    tokenized_examples['topic_id'] = []
    inaccurate = 0
    for i, offsets in enumerate(tqdm(offset_mapping)):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answer = dataset_dict['answer'][sample_index]
        # Start/end character index of the answer in the text.
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        tokenized_examples['id'].append(dataset_dict['id'][sample_index])
        tokenized_examples['topic_id'].append(dataset_dict['topic_id'][sample_index])
        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)
            # assertion to check if this checks out
            context = dataset_dict['context'][sample_index]
            offset_st = offsets[tokenized_examples['start_positions'][-1]][0]
            offset_en = offsets[tokenized_examples['end_positions'][-1]][1]
            if context[offset_st : offset_en] != answer['text'][0]:
                inaccurate += 1

    total = len(tokenized_examples['id'])
    print(f"Preprocessing not completely accurate for {inaccurate}/{total} instances")
    return tokenized_examples

def read_and_process(args, tokenizer, dataset_dict, dir_name, dataset_name, split):
    #TODO: cache this if possible
    cache_path = f'{dir_name}/{dataset_name}_encodings.pt'
    if os.path.exists(cache_path) and not args["recompute_features"]:
        tokenized_examples = util.load_pickle(cache_path)
    else:
        if split=='train':
            tokenized_examples = prepare_train_data(dataset_dict, tokenizer)
        else:
            tokenized_examples = prepare_eval_data(dataset_dict, tokenizer)
        util.save_pickle(tokenized_examples, cache_path)
    return tokenized_examples

#TODO: use a logger, use tensorboard
class Trainer():
    def __init__(self, args, log,
                 # Don't pass in this argument if you want to run without a domain-adversarial loss term
                 discriminator: Optional[DomainDiscriminator] = None):
        self.lr = args["lr"]
        self.num_epochs = args["num_epochs"]
        self.wd = args["adam_weight_decay"]
        self.discriminator_lr = args["discriminator_lr"]
        self.discriminator_momentum = args["discriminator_momentum"]

        self.device = args["device"]
        self.eval_every = args["eval_every"]
        self.path = os.path.join(args["save_dir"], 'checkpoint')
        self.num_visuals = args["num_visuals"]
        self.save_dir = args["save_dir"]
        self.log = log
        self.visualize_predictions = args["visualize_predictions"]
        self.discriminator = discriminator
        self.num_domains = 20  # TODO: We should be able set this to a meaningful number based on the data
        self.adv_loss_weight = .01
        self.nll_weights = None  # This will be initialized later
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def init_nll_weights(self, train_dataloader):
        weights_numpy = np.zeros((self.num_domains,))
        dataset_weights = train_dataloader.dataset.topic_weights()
        for i in range(len(dataset_weights)):
            weights_numpy[i%self.num_domains] += dataset_weights[i]
        weights_numpy /= self.num_domains
        self.nll_weights = torch.Tensor(weights_numpy).to(self.device)

    def save(self, model):
        model.save_pretrained(self.path)

    def evaluate(self, model, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device

        model.eval()
        pred_dict = {}
        all_start_logits = []
        all_end_logits = []
        qa_loss_sum = 0.
        num_batches = 0
        with torch.no_grad(), \
                tqdm(total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_size = len(input_ids)
                if split == 'validation':
                    outputs = model(input_ids, attention_mask=attention_mask,
                                    start_positions=batch['start_positions'].to(device),
                                    end_positions=batch['end_positions'].to(device),
                                    output_hidden_states=True)
                    # Forward
                    start_logits, end_logits = outputs.start_logits, outputs.end_logits
                    qa_loss_sum += outputs.loss.item()

                    if self.discriminator is not None:
                        hidden_cls = outputs.hidden_states[6][:, 0, :].to(device)

                        discrim_log_prob = self.discriminator(hidden_cls).to(device)
                        targets = torch.ones_like(discrim_log_prob) * (1 / self.num_domains)
                        kl_criterion = torch.nn.KLDivLoss(reduction="batchmean")
                        adv_loss = kl_criterion(discrim_log_prob, targets)

                        qa_loss_sum += self.adv_loss_weight * adv_loss.item()

                else:
                    outputs = model(input_ids, attention_mask=attention_mask)
                    # Forward
                    start_logits, end_logits = outputs.start_logits, outputs.end_logits
                # TODO: log information about adversarial network behavior?

                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                progress_bar.update(batch_size)
                num_batches += 1

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        preds = util.postprocess_qa_predictions(data_dict,
                                                 data_loader.dataset.encodings,
                                                 (start_logits, end_logits))
        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM']),
                            ('Composite QA loss', qa_loss_sum/num_batches)]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results

    def train(self, model, train_dataloader, eval_dataloader, val_dict, report=False):
        device = self.device
        model.to(device)
        if self.discriminator is not None:
            self.discriminator.to(device)
        optim = AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)
        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        if self.nll_weights is None:
            self.init_nll_weights(train_dataloader)

        if self.discriminator is not None:
            # TODO: use different learning rate for the discriminator?
            # TODO: does weight decay for the discriminator really make sense?
            discrim_optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=self.discriminator_lr, momentum=self.discriminator_momentum)

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:

                    # TODO: are we going to get weird effects training the discriminator AND the QA model on the same batch?
                    # TODO: how do we coordinate the training of the discriminator?
                    # in the paper, they use a running average loss, but I'm using AdamW here.

                    optim.zero_grad()
                    model.train()

                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)

                    outputs = model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions, output_hidden_states=True)
                    if self.discriminator is not None:
                        hidden_cls = outputs.hidden_states[6][:, 0, :].to(device)   # TODO: verify this is actually the right layer

                        labels = batch['topic_id'].to(device)
                        labels = torch.remainder(labels, self.num_domains)  # TODO: This is a really crude clustering method, improve?
                        log_prob = self.discriminator(hidden_cls).to(device)
                        targets = torch.ones_like(log_prob) * (1/self.num_domains)
                        kl_criterion = torch.nn.KLDivLoss(reduction="batchmean")
                        adv_loss = kl_criterion(log_prob, targets)  # TODO: in the paper, they have an annealing schedule for this loss term (it matters less during later epochs)
                        loss = outputs[0] + self.adv_loss_weight * adv_loss
                    else:
                        loss = outputs[0]
                    loss.backward()
                    optim.step()

                    if self.discriminator is not None:
                        # As in the paper, we're going to train the discriminator exactly the same number of times as the model, once after every epoch.
                        discrim_optimizer.zero_grad()
                        # as in the paper's github repo, we run the QA model again, with extracting new log_prob scores
                        outputs = model(input_ids, attention_mask=attention_mask,
                                        start_positions=start_positions,
                                        end_positions=end_positions, output_hidden_states=True)
                        hidden_cls = outputs.hidden_states[6][:, 0, :]  # TODO: verify this is actually the right layer
                        log_prob = self.discriminator(hidden_cls)

                        discrim_loss = self.get_discriminator_loss(log_prob, labels)

                        # TODO: track some statistic about discriminator accuracy?

                        discrim_loss.backward()
                        discrim_optimizer.step()

                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item())
                    tbx.add_scalar('train/NLL', loss.item(), global_idx)
                        
                    if self.discriminator is not None:
                        # This loss should hypothetically start low, improve, and then get worse.
                        tbx.add_scalar('train/discrim_loss', discrim_loss.item(), global_idx)

                        # This loss should hypothetically start high, get worse, and then get lower.
                        tbx.add_scalar('train/discrim_kl_div', adv_loss.item(), global_idx)

                    if report: # report performance back to tune
                        tune.report(loss=loss.item())
                        tune.report(discriminator_loss=discrim_loss.item())
                        tune.report(discriminator_kl_div=adv_loss.item())

                    if (global_idx % self.eval_every) == 0:
                        # TODO: add discriminator information?
                        self.log.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())

                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)

                            if report:
                                tune.report(f'val/{k}', v)

                        self.log.info(f'Eval {results_str}')
                        if self.visualize_predictions:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
                            self.save(model)
                    global_idx += 1
        return best_scores

    def get_discriminator_loss(self, log_prob, labels):
        # In the paper, they also use a running average loss for the QA model.
        return torch.nn.NLLLoss(weight=self.nll_weights)(log_prob, labels)


def get_dataset(args, datasets, data_dir, tokenizer, split_name):
    datasets = datasets.split(',')
    dataset_dict = None
    dataset_name=''
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
    data_encodings = read_and_process(args, tokenizer, dataset_dict, data_dir, dataset_name, split_name)
    return util.QADataset(data_encodings, train=(split_name=='train'), evaluation=(split_name=='validation'), test=(split_name=='test')), dataset_dict

def do_train(args, tokenizer):
    if args["tune"]: 
        # hacky workaround for ray tune
        # ray tune will change the cwd to tune_results/[current_results]/[results]
        # we need to change it back to the base directory
        os.chdir("../../../") 

    if not os.path.exists(args["save_dir"]):
        os.makedirs(args["save_dir"])
    args["save_dir"] = util.get_save_dir(args["save_dir"], args["run_name"])

    if args["use_checkpoint"]:
        checkpoint_path = os.path.join(args["save_dir"], 'checkpoint')
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
    else:
        model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

    log = util.get_logger(args["save_dir"], 'log_train')
    log.info(f'Args: {json.dumps(args, indent=4, sort_keys=True)}')
    log.info("Preparing Training Data...")
    args["device"] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    discriminator = DomainDiscriminator(20, 768)  # TODO: replace these 'magic numbers' with better param values
    trainer = Trainer(args, log, discriminator)
    train_dataset, _ = get_dataset(args, args["train_datasets"], args["train_dir"], tokenizer, 'train')

    log.info("Preparing Validation Data...")
    val_dataset, val_dict = get_dataset(args, args["train_datasets"], args["val_dir"], tokenizer, 'val')

    train_loader = DataLoader(train_dataset,
                            batch_size=args["batch_size"],
                            sampler=RandomSampler(train_dataset))
    val_loader = DataLoader(val_dataset,
                            batch_size=args["batch_size"],
                            sampler=SequentialSampler(val_dataset))
    best_scores = trainer.train(model, train_loader, val_loader, val_dict, report=args["tune"])

def do_eval(args, tokenizer):
    args["device"] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    split_name = 'test' if 'test' in args["eval_dir"] else 'validation'
    log = util.get_logger(args["save_dir"], f'log_{split_name}')

    trainer = Trainer(args, log)
    checkpoint_path = os.path.join(args["save_dir"], 'checkpoint')
    model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
    model.to(args["device"])

    eval_dataset, eval_dict = get_dataset(args, args["eval_datasets"], args["eval_dir"], tokenizer, split_name)
    eval_loader = DataLoader(eval_dataset,
                                batch_size=args["batch_size"],
                                sampler=SequentialSampler(eval_dataset))
    eval_preds, eval_scores = trainer.evaluate(model, eval_loader,
                                                eval_dict, return_preds=True,
                                                split=split_name)

    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())
    log.info(f'Eval {results_str}')

    # Write submission file
    sub_path = os.path.join(args["save_dir"], split_name + '_' + args["sub_file"])
    log.info(f'Writing submission file to {sub_path}...')
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(eval_preds):
            csv_writer.writerow([uuid, eval_preds[uuid]])

def main():
    # define parser and arguments
    args = get_train_test_args()

    util.set_seed(args["seed"])
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    if args["do_train"]:
        do_train(args, tokenizer)
    if args["do_eval"]:
        do_eval(args, tokenizer)


if __name__ == '__main__':
    main()
