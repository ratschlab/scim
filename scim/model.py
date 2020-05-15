import tensorflow as tf


class VAE(tf.keras.Model):
    def __init__(self,
                 latent_dim, data_dim,
                 encoder_net, decoder_net,
                 beta=1, **kwargs
                 ):

        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.data_dim = data_dim

        self.encoder_net = encoder_net
        self.decoder_net = decoder_net

        self.beta = beta

    def encode(self, inputs):
        logits = self.encoder_net(inputs)
        mu, logvar = tf.split(logits, 2, axis=1)
        return mu, logvar

    def decode(self, codes):
        return self.decoder_net(codes)

    def reparam(self, mu, logvar):
        eps = tf.random.normal(logvar.shape)
        return mu + tf.math.exp(logvar / 2) * eps

    def forward(self, inputs):
        mu, logvar = self.encode(inputs)
        codes = self.reparam(mu, logvar)
        recons = self.decode(codes)
        return recons, codes, (mu, logvar)

    def call(self, inputs):
        recons, codes, (mu, logvar) = self.foward(inputs)

        mse = tf.keras.losses.MSE(inputs, recons)
        kl = tf.math.exp(logvar) + mu**2 - logvar - 1
        kl = 0.5 * tf.reduce_sum(kl, 1)
        loss = mse + self.beta * kl

        return loss, (mse, kl), (codes, recons)


class Integrator(VAE):
    def __init__(self,
                 latent_dim, data_dim,
                 encoder_net, decoder_net,
                 discriminator, is_source_tech,
                 beta=1,
                 **kwargs
                 ):
        super(Integrator, self).__init__(latent_dim, data_dim,
                                         encoder_net, decoder_net,
                                         beta, **kwargs)
        self.discriminator = discriminator
        self.is_source_tech = is_source_tech
        return

    def adversarial_loss(self, inputs, labels=None):
        recons, codes, (mu, logvar) = self.forward(inputs)

        mse = tf.keras.losses.MSE(inputs, recons)
        adv = self.discriminator.fool(codes,
                                      labels=labels,
                                      is_source=self.is_source_tech)

        loss = mse + self.beta * adv
        return loss, (mse, adv)

    def discriminator_loss(self,
                           codes, prior,
                           code_label=None,
                           prior_label=None):
        '''Discriminator correctly classifies source/code'''

        data_loss = self.discriminator.feed(codes,
                                            labels=code_label,
                                            is_source=self.is_source_tech)

        prior_loss = self.discriminator.feed(prior,
                                             labels=prior_label,
                                             is_source=not self.is_source_tech)

        return data_loss + prior_loss
