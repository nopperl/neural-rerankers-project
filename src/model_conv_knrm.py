from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention

from model_knrm import get_mask


def conv_block(h_grams, embedding_dim, out_size):
    """A 1D-conv CNN with h-gram windows"""
    return nn.Sequential(
            nn.ConstantPad1d((0, h_grams - 1), 0),  # pad sequence in order for Conv1d to work properly
            nn.Conv1d(kernel_size=h_grams,
            in_channels=embedding_dim,
            out_channels=out_size),
            nn.ReLU())


class Conv_KNRM(nn.Module):
    '''
    Paper: Convolutional Neural Networks for Soſt-Matching N-Grams in Ad-hoc Search, Dai et al. WSDM 18
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 n_grams:int,
                 n_kernels: int,
                 conv_out_dim:int):

        super(Conv_KNRM, self).__init__()

        self.word_embeddings = word_embeddings

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)

        # todo
        self.n_grams = n_grams
        self.n_kernels = n_kernels
        self.matrix_attention = CosineMatrixAttention()
        convs = [conv_block(h, self.word_embeddings.get_output_dim(), conv_out_dim) for h in range(1, n_grams + 1)]  # a 1-D conv is applied for each h-gram where h \in [1, n]
        self.convs = nn.ModuleList(self.convs)
        self.fc = nn.Linear(n_kernels * n_grams * n_grams, 1)  # final linear model takes cross match result as input, which is a score for each kernel for each h-gram combination
        
        

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:

        #
        # prepare embedding tensors
        # -------------------------------------------------------

        # we assume 0 is padding - both need to be removed
        # shape: (batch, query_max)
        query_pad_mask = (query["tokens"] > 0).float() # > 1 to also mask oov terms
        # shape: (batch, doc_max)
        document_pad_mask = (document["tokens"] > 0).float()

        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings(query)
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings(document)

        #todo
        self.mu = self.mu.to(document_embeddings.device)
        self.sigma = self.sigma.to(query_embeddings.device)
        query_embeddings = query_embeddings.transpose(1, 2)  # embeddings must have size (batches,emb_dim,query_max)
        document_embeddings = document_embeddings.transpose(1, 2)
        h_grams_query = [conv(query_embeddings).transpose(1, 2) for conv in self.convs]  # Compute 1D-conv embedding for each h-gram
        h_grams_document = [conv(document_embeddings).transpose(1, 2) for conv in self.convs]  # query and document use the same weights
        
        # cross match result will contain concatenated vectors of (n_kernels,) for each h-gram combination, thus the resulting size will be (batches,n_kernels * n_grams * n_grams)
        cross_match = torch.empty((query_embeddings.size()[0], self.n_kernels * self.n_grams * self.n_grams), device=query_embeddings.device)
        for q in range(self.n_grams):
            for d in range(self.n_grams):
                match_matrix = self.matrix_attention(h_grams_query[q], h_grams_document[d])
                match_matrix_mask = get_mask(query_pad_mask, document_pad_mask)
                match_matrix = match_matrix * match_matrix_mask
                match_matrix = match_matrix.unsqueeze(-1)  # reshape match matrix s.t. subtraction with (n_kernels,) sized self.mu is not applied to doc dim
                rbf = torch.exp(-torch.square(match_matrix - self.mu) / (2 * torch.square(self.sigma))) # rbf.size() -> (batches,query_len,doc_len,n_kernels) due to sizes of self.mu and self.sigma
                rbf = rbf * match_matrix_mask.unsqueeze(-1)  # match_matrix_mask does not include the fourth dimension yet (rbf.size()[3] -> n_kernels)

                query_kernels = rbf.sum(dim=2)  # sum along doc dimension, query_kernels.size() -> (batches,query_len,n_kernels)
                log_query_kernels = torch.log(torch.max(query_kernels, torch.tensor(1e-10).to(query_kernels.device))) * 0.01  # take log and cap low values at 1e-10, scale results as in original implementation: https://github.com/AdeDZY/K-NRM/blob/master/knrm/model/model_knrm.py
                loq_query_kernels = log_query_kernels * query_pad_mask.unsqueeze(-1)  # query_pad_oov_mask does not include third dimension yet
                kernels = log_query_kernels.sum(dim=1)  # sum along query dimension
                offset = q * self.n_grams + d  # location in cross match tensor
                cross_match[:,offset:offset + self.n_kernels] = kernels
        
        score = self.fc(cross_match)
        output = score

        return output

    def kernel_mus(self, n_kernels: int):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """
        l_mu = [1.0]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    def kernel_sigmas(self, n_kernels: int):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :param lamb:
        :param use_exact:
        :return: l_sigma, a list of simga
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma