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

    def call(self, inputs, beta=None):
        beta = self.beta if beta is None else beta
        recons, codes, (mu, logvar) = self.forward(inputs)

        mse = tf.keras.losses.MSE(inputs, recons)
        kl = tf.math.exp(logvar) + mu**2 - logvar - 1
        kl = 0.5 * tf.reduce_sum(kl, 1)
        loss = mse + beta * kl

        return loss, (mse, kl), (codes, recons)


class Integrator(VAE):
    def __init__(self,
                 latent_dim, data_dim,
                 encoder_net, decoder_net,
                 discriminator, is_source,
                 beta=1,
                 **kwargs
                 ):

        super(Integrator, self).__init__(latent_dim, data_dim,
                                         encoder_net, decoder_net,
                                         beta, **kwargs)

        # source tech is used as discriminator positive class
        self.is_source = is_source
        self.discriminator = discriminator


        return

    def generator_variables(self):
        return self.encoder_net.trainable_variables \
               + self.decoder_net.trainable_variables

    def call(self, *args, **kwargs):
        raise NotImplementedError

    def vae(self, inputs, beta=None):
        return super().call(inputs, beta)

    def adversarial_loss(self, inputs, labels=None):
        recons, codes, (mu, logvar) = self.forward(inputs)

        mse = tf.keras.losses.MSE(inputs, recons)

        # fool the discriminator
        adv = self.discriminator.loss(codes,
                                      labels=labels,
                                      real=not self.is_source)

        loss = mse + self.beta * adv
        return loss, (mse, adv)

    def discriminator_loss(self,
                           codes, prior,
                           code_label=None,
                           prior_label=None):
        '''Discriminator correctly classifies technology'''

        data_loss = self.discriminator.loss(codes,
                                            labels=code_label,
                                            real=self.is_source)

        prior_loss = self.discriminator.loss(prior,
                                             labels=prior_label,
                                             real=not self.is_source)

        return data_loss + prior_loss
