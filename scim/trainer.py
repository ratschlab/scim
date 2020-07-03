import tensorflow as tf


class Trainer:
    def __init__(self, encoder_lut, decoder_lut,
                 data_dim_lut, latent_dim,
                 discriminator, source_key,
                 discopt,
                 genopt_lut,
                 beta):

        techs = data_dim_lut.keys()
        assert encoder_lut.keys() == techs
        assert decoder_lut.keys() == techs
        assert genopt_lut.keys() == techs

        self.model = dict()
        for key in techs:
            self.model[key] = Integrator(
                    latent_dim=latent_dim,
                    data_dim=data_dim_lut[key],
                    encoder_net=encoder_lut[key],
                    decoder_net=decoder_lut[key],
                    discriminator=discriminator,
                    beta=beta,
                    is_source=key == source_key
                    )

        self.discriminator = discriminator
        self.discopt = discopt
        self.genopt_lut = genopt_lut
        return

    def vae_step(self, inputs, tech, beta=None):
        with tf.GradientTape() as tape:
            loss, (mse, kl), (codes, recon) = self.model[tech].vae(inputs, beta=beta)
        tvs = self.model[tech].generator_variables()
        grads = tape.gradient(loss, tvs)
        self.genopt[key].apply_gradients(zip(grads, tvs))
        return loss, (mse, kl), (codes, recon)

