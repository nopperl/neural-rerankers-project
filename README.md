# Team information

Student 1 Matrikelnummer + Name:

# Report

In general, we aimed to leverage as much built-in Pytorch and AlllenNLP functions as possible. The biggest problem we faced when trying the models was a bug in the metrics calculation (`core_metrics.calculate_metrics_plain()`). We found out in the discussion forum that query and doc ids have to be passed as strings. This led to confusion about the performance of our models on our part since we only found out about that late in the process. In our view, it would have been nice to at least type annotate the function, as is the case with the model classes.

We also inadverdently implemented our own `core_metrics.unrolled_to_ranked_result()` function, since we didn't discover this function until it was too late. The function can be seen in `train.py` and is probably a bit more efficient since it does not evaluate the whole iterator.

For our optimizer we used Adam with default values and MarginRankingLoss as a loss function. We decided to use `mean` instead of `elementwise_mean`, since the latter does not seem to be supported anymore.

All models were trained on a GPU with early stopping after one epoch of no improvement.

## MatchPyramid

The MatchPyramid model interprets the match matrix of query and document as image and applies a standard image deep learning pipeline (i.e. convolutional layers).
Since we did not implement dynamic pooling before, we figured we would have the most problems with this part. As it turns out, however, it is pretty easy to implement these layers in Pytorch.
We took care to make the architecture fully parameterizable (by the three given parameters). TThe output shape of the convolutional part can be calculated solely by the given parameters.
There remains a fixed parameter in the MLP part however: The first fully connected layer outputs 300 units. This number is chosen rather arbitrarily and might not even be necessary. In our experiments it led to a slightly better accuracy at next to no runtime gains.
We chose the `allennlp.modules.matrix_attention.cosine_matrix_attention.CosineMatrixAttention` to calculate the match matrix.

We trained the MatchPyramid for two epochs over the whole MS-MARCO train dataset, which took 41:19 mins per epoch. The average loss at the end of the training amounted to 0.03. We then computed the MRR@10 over the MS-MARCO and FiRA test datasets and got passable results of around 0.21 and 0.98 respectively. The results are also given in the table below. For completeness, other metrics are also given below.

| Model | Training time | Avg Loss | MS-MARCO MRR@10 | FiRA MRR@10 |
| --- | --- | --- | --- | --- |
| MatchPyramid | 1,025 ms/sample | 0.03 | 0.21 | 0.98 |

MS-MARCO:
```
{'MRR@10': 0.21137242063492065,
 'Recall@10': 0.43129166666666663,
 'QueriesWithNoRelevant@10': 1124,
 'QueriesWithRelevant@10': 876,
 'AverageRankGoldLabel@10': 3.6780821917808217,
 'MedianRankGoldLabel@10': 3.0,
 'MRR@20': 0.21837864527328538,
 'Recall@20': 0.53175,
 'QueriesWithNoRelevant@20': 923,
 'QueriesWithRelevant@20': 1077,
 'AverageRankGoldLabel@20': 5.773444753946147,
 'MedianRankGoldLabel@20': 4.0,
 'MRR@1000': 0.2209044264145279,
 'Recall@1000': 0.6002916666666668,
 'QueriesWithNoRelevant@1000': 788,
 'QueriesWithRelevant@1000': 1212,
 'AverageRankGoldLabel@1000': 8.202145214521453,
 'MedianRankGoldLabel@1000': 5.0,
 'nDCG@3': 0.19586360483641874,
 'nDCG@5': 0.22510904998988543,
 'nDCG@10': 0.2615518413594442,
 'nDCG@20': 0.28723964719582057,
 'nDCG@1000': 0.30169530966266495,
 'QueriesRanked': 2000,
 'MAP@1000': 0.21737391977083997}
```

FiRA:
```
{'MRR@10': 0.9767441860465116,
 'Recall@10': 0.17404415905677123,
 'QueriesWithNoRelevant@10': 0,
 'QueriesWithRelevant@10': 43,
 'AverageRankGoldLabel@10': 1.0465116279069768,
 'MedianRankGoldLabel@10': 1.0,
 'MRR@20': 0.9767441860465116,
 'Recall@20': 0.2611591616075389,
 'QueriesWithNoRelevant@20': 0,
 'QueriesWithRelevant@20': 43,
 'AverageRankGoldLabel@20': 1.0465116279069768,
 'MedianRankGoldLabel@20': 1.0,
 'MRR@1000': 0.9767441860465116,
 'Recall@1000': 0.9524313968713792,
 'QueriesWithNoRelevant@1000': 0,
 'QueriesWithRelevant@1000': 43,
 'AverageRankGoldLabel@1000': 1.0465116279069768,
 'MedianRankGoldLabel@1000': 1.0,
 'nDCG@3': 0.7243909354176328,
 'nDCG@5': 0.7015509858943222,
 'nDCG@10': 0.6987151191614527,
 'nDCG@20': 0.7054590011324382,
 'nDCG@1000': 0.8630575644969966,
 'QueriesRanked': 43,
 'MAP@1000': 0.7806980198477049}
```
