from typing import Dict, Iterator, List,Tuple

import torch
import torch.nn as nn                            
import torch.nn.functional as F

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention

def block(in_size, out_size, kernel_size, pooling_size):
    """A standardized convolutional block with a 2d convolution, relu activation and adaptive pooling"""
    return nn.Sequential(
        nn.Conv2d(in_size, out_size, kernel_size),
        nn.ReLU(),
        nn.AdaptiveMaxPool2d(pooling_size)
    )

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
        self.min_length = max(conv_kernel_size[0])
        self.matrix_attention = CosineMatrixAttention()

        conv_input_size = conv_output_size.copy()
        conv_input_size[0] = 1  # conv in channels are the out channels of the preceding layers (first layer has a single channel as input)
        self.convs = nn.Sequential(*[block(*params) for params in zip(conv_input_size, conv_output_size , conv_kernel_size, adaptive_pooling_size)])

        out_size = conv_output_size[-1] * adaptive_pooling_size[-1][0] * adaptive_pooling_size[-1][1]  # final output layer can be calculated beforehand thanks to adaptive pooling
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_size, 300),  # fully connected layer to flattened conv output
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
        query_len = query_embeddings.size()[1]
        doc_len = document_embeddings.size()[1]
        if query_len < self.min_length:  # Pad embeddings if the embedded sequence is smaller than the conv kernel
            query_embeddings = F.pad(query_embeddings, pad=(0, 0, 0, self.min_length - query_len))
        if doc_len < self.min_length:
            doc_embeddings = F.pad(doc_embeddings, pad=(0, 0, 0, self.min_length - doc_len))
        match_matrix = self.matrix_attention(query_embeddings, document_embeddings)  # calculate match matrix using efficient attention operation
        match_size = match_matrix.size()
        match_matrix = match_matrix.view(match_size[0], 1, match_size[1], match_size[2])  # reshape tensor to the channel format required by conv layers
        features = self.convs(match_matrix)
        scores = self.mlp(features)
        output = scores

        return output
