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
We took care to make the architecture fully parameterizable (by the three given parameters). The output shape of the convolutional part can be calculated solely by the given parameters.
There remains a fixed parameter in the MLP part however: The first fully connected layer outputs 300 units. This number is chosen rather arbitrarily and might not even be necessary. In our experiments it led to a slightly better accuracy at next to no runtime gains.
We chose the `allennlp.modules.matrix_attention.cosine_matrix_attention.CosineMatrixAttention` to calculate the match matrix.

We trained the MatchPyramid for two epochs over the whole MS-MARCO train dataset, which took on average 41:19 mins per epoch. The average loss at the end of the training amounted to 0.03. We then computed the MRR@10 over the MS-MARCO and FiRA test datasets and got passable results of around 0.21 and 0.98 respectively. The complete results on the test sets are given below.

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

## KNRM

The KNRM model generates a fixed number of bins by applying learnable kernels to the match matrix. The resulting score is calculated using a linear model.

Again, we choose the `allennlp.modules.matrix_attention.cosine_matrix_attention.CosineMatrixAttention` to calculate the match matrix.

The most significant problem we faced were extremely low values when calculating $\log(K_k(M_i))$. This lead to unusable `nan` results. We found out when checking the [original implementation][1] that they solved this by capping low values to 1e-10 and scaling the values by 0.01. This is not described in the [paper][2], however. We fixed the problem in our model likewise.

Another implementation detail is how to compute the OOV mask for the match matrix. We reimplemented the `get_mask()` function of the [original implementation][1] in Pytorch. Since it seems to be only a matrix multiplication, we implemented it using the more efficient `torch.bmm()` instead of nested for loops.

We trained the KNRM for two epochs over the whole MS-MARCO train dataset, which took on average 32:46 mins per epoch. This clearly shows that KNRM is faster than MatchPyramid. The average loss at the end of the training amounted to 0.04. We then computed the MRR@10 over the MS-MARCO and FiRA test datasets and got passable results of around 0.22 and 0.97 respectively. The complete results on the test sets are given below.

MS-MARCO:
```
{'MRR@10': 0.22052698412698413,
 'Recall@10': 0.4628333333333333,
 'QueriesWithNoRelevant@10': 1061,
 'QueriesWithRelevant@10': 939,
 'AverageRankGoldLabel@10': 3.7380191693290734,
 'MedianRankGoldLabel@10': 3.0,
 'MRR@20': 0.22665102423376418,
 'Recall@20': 0.55125,
 'QueriesWithNoRelevant@20': 884,
 'QueriesWithRelevant@20': 1116,
 'AverageRankGoldLabel@20': 5.514336917562724,
 'MedianRankGoldLabel@20': 4.0,
 'MRR@1000': 0.22841715720041084,
 'Recall@1000': 0.6002916666666668,
 'QueriesWithNoRelevant@1000': 788,
 'QueriesWithRelevant@1000': 1212,
 'AverageRankGoldLabel@1000': 7.298679867986799,
 'MedianRankGoldLabel@1000': 4.0,
 'nDCG@3': 0.20433524401045344,
 'nDCG@5': 0.23704505274893858,
 'nDCG@10': 0.2762912017386188,
 'nDCG@20': 0.29887043609675706,
 'nDCG@1000': 0.3092384406034222,
 'QueriesRanked': 2000,
 'MAP@1000': 0.22529540687041122}
```

FiRA:
```
{'MRR@10': 0.9728682170542636,
 'Recall@10': 0.17626463120655503,
 'QueriesWithNoRelevant@10': 0,
 'QueriesWithRelevant@10': 43,
 'AverageRankGoldLabel@10': 1.069767441860465,
 'MedianRankGoldLabel@10': 1.0,
 'MRR@20': 0.9728682170542636,
 'Recall@20': 0.2585596782510704,
 'QueriesWithNoRelevant@20': 0,
 'QueriesWithRelevant@20': 43,
 'AverageRankGoldLabel@20': 1.069767441860465,
 'MedianRankGoldLabel@20': 1.0,
 'MRR@1000': 0.9728682170542636,
 'Recall@1000': 0.9541506641086419,
 'QueriesWithNoRelevant@1000': 0,
 'QueriesWithRelevant@1000': 43,
 'AverageRankGoldLabel@1000': 1.069767441860465,
 'MedianRankGoldLabel@1000': 1.0,
 'nDCG@3': 0.6867852002191022,
 'nDCG@5': 0.6881318980167349,
 'nDCG@10': 0.6957115295267123,
 'nDCG@20': 0.7089894487448578,
 'nDCG@1000': 0.8643313057286531,
 'QueriesRanked': 43,
 'MAP@1000': 0.7948408237588007}
```

## Conv-KNRM

The Conv-KNRM model applies 1D CNNs with different window sizes on the query and document embeddings. The resulting h-gram embeddings of the query and the document are then cross matched using the same procedure as in KNRM. A final linear model computes the score using the concatenated cross match kernels.

The 1D convolutional computation was no issue for us, although we were initially unsure whether query and document have separate weights (they do not). For the cross match procedure, we simply copied the KNRM implementation.

The greatest potential for a runtime performance improvement is probably the cross match computation, which iterates through all h-gram combinations and sequentially computes the specific kernel outputs. There may be a way to do this in parallel or at least in a more efficient manner. We tried to optimize it as much as possible, but there might still be possible improvements.

We trained the Conv-KNRM for two epochs over the whole MS-MARCO train dataset, which took on average 55:38 mins per epoch. This makes Conv-KNRM the slowest of all models, which is clearly due to the sequential computation. Unfortunately, due to this long time, we were not able to train the model fully, reaching only an MRR@10 on MS-MARCO and FiRA of 0.20 and 0.93 respectively. The complete results on the test sets are given below.

MS-MARCO:
```
{'MRR@10': 0.19505158730158728,
 'Recall@10': 0.39995833333333336,
 'QueriesWithNoRelevant@10': 1188,
 'QueriesWithRelevant@10': 812,
 'AverageRankGoldLabel@10': 3.666256157635468,
 'MedianRankGoldLabel@10': 3.0,
 'MRR@20': 0.20192665926050213,
 'Recall@20': 0.49824999999999997,
 'QueriesWithNoRelevant@20': 991,
 'QueriesWithRelevant@20': 1009,
 'AverageRankGoldLabel@20': 5.851337958374629,
 'MedianRankGoldLabel@20': 4.0,
 'MRR@1000': 0.20553093883520332,
 'Recall@1000': 0.6002916666666668,
 'QueriesWithNoRelevant@1000': 788,
 'QueriesWithRelevant@1000': 1212,
 'AverageRankGoldLabel@1000': 9.766501650165017,
 'MedianRankGoldLabel@1000': 5.0,
 'nDCG@3': 0.17892262546055587,
 'nDCG@5': 0.21012103702834278,
 'nDCG@10': 0.24226767235717317,
 'nDCG@20': 0.2674499135592376,
 'nDCG@1000': 0.2885970890570177,
 'QueriesRanked': 2000,
 'MAP@1000': 0.2029869699517343}
```

FiRA:
```
{'MRR@10': 0.9263565891472869,
 'Recall@10': 0.15682592176501056,
 'QueriesWithNoRelevant@10': 0,
 'QueriesWithRelevant@10': 43,
 'AverageRankGoldLabel@10': 1.1627906976744187,
 'MedianRankGoldLabel@10': 1.0,
 'MRR@20': 0.9263565891472869,
 'Recall@20': 0.2422370568194677,
 'QueriesWithNoRelevant@20': 0,
 'QueriesWithRelevant@20': 43,
 'AverageRankGoldLabel@20': 1.1627906976744187,
 'MedianRankGoldLabel@20': 1.0,
 'MRR@1000': 0.9263565891472869,
 'Recall@1000': 0.9434590879921583,
 'QueriesWithNoRelevant@1000': 0,
 'QueriesWithRelevant@1000': 43,
 'AverageRankGoldLabel@1000': 1.1627906976744187,
 'MedianRankGoldLabel@1000': 1.0,
 'nDCG@3': 0.5671777793002863,
 'nDCG@5': 0.5714586851605508,
 'nDCG@10': 0.5693508654263322,
 'nDCG@20': 0.574296199194637,
 'nDCG@1000': 0.8056777798492057,
 'QueriesRanked': 43,
 'MAP@1000': 0.7105657304955493}
```

## Results

| Model | Training time in ms/sample | Avg Loss | MS-MARCO MRR@10 | FiRA MRR@10 |
| --- | --- | --- | --- | --- |
| MatchPyramid | 1,025 | 0.03 | 0.21 | 0.98 |
| KNRM | 0.822 | 0.04 | 0.22 | 0.97 |
| Conv-KNRM | 1.420 | 0.06 | 0.20  | 0.93  |


[1]: https://github.com/AdeDZY/K-NRM
[2]: https://www.cs.cmu.edu/~zhuyund/papers/end-end-neural.pdf
