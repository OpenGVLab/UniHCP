import torch
import torch.nn as nn
import torch.nn.functional as F


class Norm2d(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps
        self.normalized_shape = (embed_dim,)

        #  >>> workaround for compatability
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln.weight = self.weight
        self.ln.bias = self.bias

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def _get_activation(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class SimpleNeck(nn.Module):
    def __init__(self,
                 mask_dim,
                 backbone,  # placeholder
                 bn_group,
                 activation='gelu',
                 task_sp_list=(),
                 mask_forward=True
                ):
        super(SimpleNeck, self).__init__()
        self.task_sp_list = task_sp_list

        self.vis_token_dim = self.embed_dim = backbone.embed_dim
        self.mask_dim = mask_dim

        self.mask_map = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
            Norm2d(self.embed_dim),
            _get_activation(activation),
            nn.ConvTranspose2d(self.embed_dim, self.mask_dim, kernel_size=2, stride=2),
        ) if mask_dim else False

        self.maskformer_num_feature_levels = 1  # always use 3 scales

        self.mask_forward = mask_forward

    def forward(self, features):
        if self.mask_map and self.mask_forward:
            features.update({'neck_output': {'mask_features': self.mask_map(features['backbone_output']),
                                             'multi_scale_features': [features['backbone_output']]}})
        else:
            features.update({'neck_output': {'mask_features': None,
                                             'multi_scale_features': [features['backbone_output']]}})
        return features


