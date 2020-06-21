from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment
prepare_environment(Params({})) # sets the seeds to be fixed

import torch

from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

from data_loading import *
from model_knrm import *
from model_conv_knrm import *
from model_match_pyramid import *

from datetime import datetime
from math import isnan
import numpy as np
from operator import itemgetter

# change paths to your data directory
config = {
    "vocab_directory": "../data/allen_vocab_lower_10",
    "pre_trained_embedding": "../data/glove.42B.300d.txt",
    "model": "match_pyramid",
    "train_data":"../data/triples.train.tsv",
    "validation_data":"../data/msmarco_tuples.validation.tsv",
    "test_data":"../data/msmarco_tuples.test.tsv",
    "fira_test_data":"../data/fira_numsnippets_test_tuples.tsv",
    "qrels":"../data/msmarco_qrels.txt",
    "fira_qrels":"../data/fira_numsnippets_qrels.txt",
    "early_stopping": 1
}
config["model_file"] = f"../models/{config['model']}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

#
# data loading
#

vocab = Vocabulary.from_files(config["vocab_directory"])
tokens_embedder = Embedding.from_params(vocab, Params({"pretrained_file": config["pre_trained_embedding"],
                                                      "embedding_dim": 300,
                                                      "trainable": True,
                                                      "padding_index":0}))

word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})

# recommended default params for the models (but you may change them if you want)
if config["model"] == "knrm":
    model = KNRM(word_embedder, n_kernels=11)
elif config["model"] == "conv_knrm":
    model = Conv_KNRM(word_embedder, n_grams=3, n_kernels=11, conv_out_dim=128)
elif config["model"] == "match_pyramid":
    model = MatchPyramid(word_embedder, conv_output_size=[16,16,16,16,16], conv_kernel_size=[[3,3],[3,3],[3,3],[3,3],[3,3]], adaptive_pooling_size=[[36,90],[18,60],[9,30],[6,20],[3,10]])


# todo optimizer, loss
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
criterion = nn.MarginRankingLoss(margin=1, reduction='mean')
optimizer = torch.optim.Adam(model.parameters())

msmarco_qrels = load_qrels(config["qrels"])

print('Model',config["model"],'total parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
print('Network:', model)

def validate(data_file, qrels):
    _tuple_loader = IrLabeledTupleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30) # not spacy tokenized already (default is spacy)
    _iterator = BucketIterator(batch_size=128,
                               sorting_keys=[("doc_tokens", "num_tokens"), ("query_tokens", "num_tokens")])
    _iterator.index_with(vocab)

    model.eval()
    scores = {}
    for batch in Tqdm.tqdm(_iterator(_tuple_loader.read(data_file), num_epochs=1)):
        # todo test loop
        # todo evaluation
        batch['query_tokens']['tokens'] = batch['query_tokens']['tokens'].to(device)
        batch['doc_tokens']['tokens'] = batch['doc_tokens']['tokens'].to(device)
        with torch.no_grad():
            batch_scores = model.forward(batch['query_tokens'], batch['doc_tokens'])

        for i in range(len(batch_scores)):
            query_id = batch['query_id'][i].item()
            if query_id not in scores:
                scores[query_id] = []
            scores[query_id].append((batch['doc_id'][i].item(), batch_scores[i].item()))

    ranking = {}
    for query_id in scores:
        ranking[query_id] = next(zip(*sorted(scores[query_id], key=itemgetter(1), reverse=True)))

    results = calculate_metrics_plain(ranking, qrels)
    return results

#
# train
#

_triple_loader = IrTripleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30,tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())) # already spacy tokenized, so that it is faster 

_iterator = BucketIterator(batch_size=64,
                           sorting_keys=[("doc_pos_tokens", "num_tokens"), ("doc_neg_tokens", "num_tokens")])

_iterator.index_with(vocab)

mrrs = []
epochs_since_max = 0
for epoch in range(2):
    model.train()
    losses = []
    for batch in Tqdm.tqdm(_iterator(_triple_loader.read(config["train_data"]), num_epochs=1)):
        # todo train loop
        optimizer.zero_grad()
        batch['query_tokens']['tokens'] = batch['query_tokens']['tokens'].to(device)
        batch['doc_pos_tokens']['tokens'] = batch['doc_pos_tokens']['tokens'].to(device)
        batch['doc_neg_tokens']['tokens'] = batch['doc_neg_tokens']['tokens'].to(device)
        relevance_pos = model.forward(batch['query_tokens'], batch['doc_pos_tokens'])
        relevance_neg = model.forward(batch['query_tokens'], batch['doc_neg_tokens'])
        labels = torch.ones(len(batch)).to(device)
        loss = criterion(relevance_pos, relevance_neg, labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    val_results = validate(config["validation_data"], msmarco_qrels)
    mrr = val_results['MRR@10']
    print(f"Epoch {epoch} Loss: {np.mean(losses)}, MRR@10: {mrr}")
    mrrs.append(mrr)
    if mrr < max(mrrs) and not isnan(max(mrrs)):
        epochs_since_max += 1
    else:
        epochs_since_max = 0
    if epochs_since_max >= config["early_stopping"]:
        print(f"Halting training due to early stopping (epochs: {config['early_stopping']})")
        break

torch.save(model.state_dict(), config["model_file"])

#
# eval (duplicate for validation inside train loop)
#

msmarco_results = validate(config["test_data"], msmarco_qrels)
print("msmarco results")
print(msmarco_results)

fira_qrels = load_qrels(config["fira_qrels"])
fira_results = validate(config["fira_test_data"], fira_qrels)

print("fira results")
print(fira_results)