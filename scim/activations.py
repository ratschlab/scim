import tensorflow as tf
from tensorflow_probability import distributions as tfd


class Deterministic:
    '''Pushes logits through a Deterministic likelihood

    This is useful for deterministic encoders.
    You can still call code.sample(), but it will just return the loc
    '''
    def __init__(self, output_dim=None):
        self.output_dim = output_dim
        self.name = 'Deterministic'

    def params_size(self):
        return self.output_dim

    def __call__(self, logits, **kwargs):
        lkl = tfd.Deterministic(loc=logits, **kwargs)
        return tfd.Independent(lkl, reinterpreted_batch_ndims=1)


class DiagonalGaussian:
    '''Pushes logits through a Gaussian likelihood.
    '''
    def __init__(self, output_dim=None):
        self.output_dim = output_dim
        self.name = 'DiagonalGaussian'

    def params_size(self):
        return 2 * self.output_dim

    def __call__(self, logits, scale_identity_multiplier=1e-4, **kwargs):
        loc = logits[..., :self.output_dim]
        scale = tf.nn.softplus(logits[..., self.output_dim:])
        lkl = tfd.MultivariateNormalDiag(
                loc=loc,
                scale_diag=scale,
                scale_identity_multiplier=scale_identity_multiplier,
                **kwargs
                )

        return lkl


class IsotropicGaussian:
    '''Pushes logits through a Gaussian likelihood.
    '''
    def __init__(self, output_dim=None):
        self.output_dim = output_dim
        self.name = 'IsotropicGaussian'

    def params_size(self):
        return self.output_dim

    def __call__(self, logits):
        loc = logits
        lkl = tfd.MultivariateNormalDiag(loc=loc)
        return lkl
