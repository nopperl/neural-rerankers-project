from typing import Dict, Iterator, List,Tuple

import torch
import torch.nn as nn                            
import torch.nn.functional as F

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention

class MatchPyramid(nn.Module):
    '''
    Paper: Text Matching as Image Recognition, Pang et al., AAAI'16
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 conv_output_size: List[int],
                 conv_kernel_size: List[Tuple[int,int]],
                 adaptive_pooling_size: List[Tuple[int,int]]):

        super(MatchPyramid, self).__init__()

        self.word_embeddings = word_embeddings

        if len(conv_output_size) != len(conv_kernel_size) or len(conv_output_size) != len(adaptive_pooling_size):
            raise Exception("conv_output_size, conv_kernel_size, adaptive_pooling_size must have the same length")

        # todo
        self.matrix_attention = CosineMatrixAttention()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 8, 5),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(50),  # dynamic pooling to fixed features
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 50 * 50, 300),  # fully connected layer to flattened conv output
            nn.ReLU(),
            nn.Linear(300, 1)  # outputs a single score value
        )

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:

        #
        # prepare embedding tensors
        # -------------------------------------------------------

        # shape: (batch, query_max)
        query_pad_oov_mask = (query["tokens"] > 0).float()
        # shape: (batch, doc_max)
        document_pad_oov_mask = (document["tokens"] > 0).float()

        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings(query)
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings(document)

        # todo
        match_matrix = self.matrix_attention(query_embeddings, document_embeddings)
        features = self.convs(match_matrix)
        scores = self.mlp(features)
        output = scores

        return output
