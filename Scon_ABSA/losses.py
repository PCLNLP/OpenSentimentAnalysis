from __future__ import print_function
from contextlib import AsyncExitStack

import mindspore
from mindspore import nn, ops, numpy

def normalization(data):

    for i in range(len(data)):

        _range = ops.ArgMaxWithValue()(data[i])[1] - ops.ArgMinWithValue()(data[i])[1]
        data[i] = (data[i] - ops.ArgMinWithValue()(data[i])[1]) / _range
    return data


class SupConLoss(nn.Cell):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='one',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def construct(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], features.shape[1], -1)

        # get batch_size
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = ops.Eye()(batch_size, mindspore.float32)
        elif labels is not None:
            labels = labels.view(-1, 1)     # 16*1
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = ops.Equal()(labels, labels.T).astype("float32")      # 16*16
        else:
            mask = mask.astype("float32")

        features = ops.ExpandDims()(features, 1)
        features = ops.L2Normalize(2)(features)
        contrast_count = features.shape[1]
        contrast_feature = ops.Concat(0)(ops.Unstack(1)(features))

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = ops.Div()(
            ops.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        _, logits_max = ops.ArgMaxWithValue(axis=1, keep_dims=True)(anchor_dot_contrast)
        logits = anchor_dot_contrast - logits_max
        #logits = anchor_dot_contrast - logits_max.detach()

        _, logits_min = ops.ArgMinWithValue(axis=1, keep_dims=True)(logits)
        _, logits_max = ops.ArgMaxWithValue(axis=1, keep_dims=True)(logits)
        _range = logits_max - logits_min
        logits = ops.Div()(logits-logits_min,_range)
        mask = numpy.tile(mask, (anchor_count, contrast_count))

        logits_mask = ops.OnesLike()(mask) - ops.Eye()(mask.shape[0], mask.shape[1], mindspore.float32)

        mask = mask * logits_mask

        exp_logits = ops.Exp()(logits) * logits_mask

        log_prob = logits - ops.Log()(exp_logits.sum(axis=1, keepdims=True))

        mean_log_prob_pos = (mask * log_prob).sum(axis=1) / (mask.sum(axis=1)+1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()


        return loss
