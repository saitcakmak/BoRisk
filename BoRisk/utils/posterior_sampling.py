import torch
from torch import Tensor
from botorch.models.model import Model
from typing import Union, Iterable, Optional
from gp_sampling import decoupled_sampler


def exact_posterior_sampling(
    X: Tensor,
    model: Model,
    sample_shape: Union[torch.Size, Iterable],
    base_samples: Optional[Tensor] = None,
) -> Tensor:
    r"""
    Draw samples from the GP posterior.

    :param X: The tensor of points to sample at
    :param model: The GP model
    :param sample_shape: The batch shape of the samples.
    :param base_samples: Base samples to pass on to rsample.
    :return: A `sample_shape x model_batch_shape x 1 x dim` tensor of posterior samples
    """
    return model.posterior(X).rsample(
        sample_shape=sample_shape, base_samples=base_samples
    )


def decoupled_posterior_sampling(
    X: Tensor,
    model: Model,
    sample_shape: Union[torch.Size, Iterable],
    base_samples: Optional[Tensor] = None,
) -> Tensor:
    r"""
    Draw approximate samples from the GP posterior using decoupled samplers.

    :param X: The tensor of points to sample at
    :param model: The GP model
    :param sample_shape: The batch shape of the samples.
    :param base_samples: Ignored.
    :return: A `sample_shape x model_batch_shape x 1 x dim` tensor of approximate
        posterior samples
    """
    sampler = decoupled_sampler(
        model=model,
        sample_shape=sample_shape,
        num_basis=256,
        input_batch_shape=X.shape[:-2],
    )
    return sampler(X)
