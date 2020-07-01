from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention


def get_mask(query_mask, doc_mask):
    """Generate oov token mask for match matrix through trick: if at least one item of the matmul is 0, the resulting item will be 0"""
    mask = torch.bmm(query_mask.unsqueeze(-1), doc_mask.unsqueeze(-1).transpose(-1, -2))
    return mask


class KNRM(nn.Module):
    '''
    Paper: End-to-End Neural Ad-hoc Ranking with Kernel Pooling, Xiong et al., SIGIR'17
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 n_kernels: int):

        super(KNRM, self).__init__()

        self.word_embeddings = word_embeddings

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)

        #todo
        self.matrix_attention = CosineMatrixAttention()
        self.fc = nn.Linear(n_kernels, 1)

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        # shape: (batch, query_max)
        query_pad_oov_mask = (query["tokens"] > 0).float() # > 1 to also mask oov terms
        # shape: (batch, doc_max)
        document_pad_oov_mask = (document["tokens"] > 0).float()

        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings(query)
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings(document)

        #todo
        self.mu = self.mu.to(document_embeddings.device)
        self.sigma = self.sigma.to(query_embeddings.device)
        match_matrix = self.matrix_attention(query_embeddings, document_embeddings)
        match_matrix_mask = get_mask(query_pad_oov_mask, document_pad_oov_mask)
        match_matrix = match_matrix * match_matrix_mask
        match_matrix = match_matrix.unsqueeze(-1)  # reshape match matrix s.t. subtraction with (n_kernels,) sized self.mu is not applied to doc dim
        rbf = torch.exp(-torch.square(match_matrix - self.mu) / (2 * torch.square(self.sigma))) # rbf.size() -> (batches,query_len,doc_len,n_kernels) due to sizes of self.mu and self.sigma
        rbf = rbf * match_matrix_mask.unsqueeze(-1)  # match_matrix_mask does not include the fourth dimension yet (rbf.size()[3] -> n_kernels)
        
        query_kernels = rbf.sum(dim=2)  # sum along doc dimension, query_kernels.size() -> (batches,query_len,n_kernels)
        log_query_kernels = torch.log(torch.max(query_kernels, torch.tensor(1e-10).to(query_kernels.device))) * 0.01  # take log and cap low values at 1e-10, scale results as in original implementation: https://github.com/AdeDZY/K-NRM/blob/master/knrm/model/model_knrm.py
        loq_query_kernels = log_query_kernels * query_pad_oov_mask.unsqueeze(-1)  # query_pad_oov_mask does not include third dimension yet
        kernels = log_query_kernels.sum(dim=1)  # sum along query dimension
        score = self.fc(kernels) # kernels.size() -> (batches,n_kernels)  (indepenent of query, doc len)
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
        l_sigma = [0.0001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma
