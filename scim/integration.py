import tensorflow.keras as tfk


class Integrator(tfk.Model):
    def __init__(
            self,
            latent_dim, data_dim,
            encoder, decoder, discriminator,
            lmbda=1,
            discriminator_input_cats=None,
            ):

        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.lmbda = lmbda
        self.posterior = ''  # TODO: add posterior
        self.likelihood = ''  # TODO: add likelihood

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
        return nll, adv
