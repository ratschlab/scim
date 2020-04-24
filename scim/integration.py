import tensorflow.keras as tfk
from . import activations as act


class Integrator(tfk.Model):
    def __init__(
            self,
            latent_dim, data_dim,
            encoder, decoder, discriminator,
            lmda=1,
            discriminator_input_cats=None,
            likelihood='IsotropicGaussian',
            posterior='IsotropicGaussian'
            ):

        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.lmda = lmda

        if isinstance(likelihood, str):
            likelihood = getattr(act, likelihood)
        self.likelihood = likelihood(self.data_dim)

        if isinstance(posterior, str):
            posterior = getattr(act, posterior)
        self.posterior = posterior(self.latent_dim)
        return

    def encode(self, data):
        '''Map inputs into latent space'''
        mapped = self.encoder(data)
        return self.posterior(mapped)

    def decode(self, code):
        '''Reconstruct inputs from latent space'''
        mapped = self.decoder(code)
        return self.likelihood(mapped)

    def generator_loss(self, codes, labels=None):
        '''Discriminator misclassifies codes as source'''
        return self.discriminator.fool(codes, 'code', labels=labels)

    def discriminator_loss(self,
                           codes, prior,
                           code_label=None,
                           prior_label=None):
        '''Discriminator correctly classifies source/code'''

        data_loss = self.discriminator.feed(
                codes, 'code',
                labels=code_label)

        prior_loss = self.discriminator.feed(
                prior, 'prior',
                labels=prior_label)

        return data_loss + prior_loss

    def adversarial_loss(self, inputs, labels=None):
        '''Reconstruct & cool discriminator'''
        codes = self.encode(inputs).sample()
        recon = self.decode(codes)

        nll = -recon.log_prob(inputs).sum(1)
        adv = self.generator_loss(codes, labels)
        loss = nll + self.model.beta * adv
        return loss, nll, adv
