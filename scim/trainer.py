import tensorflow as tf


class Trainer:
    def __init__(self, model,
                 genopt, crtopt,
                 niters_critic=5):

        self.model = model
        self.genopt = genopt
        self.crtopt = crtopt
        self.niters_critic = niters_critic

    def update_adversarial(self, inputs, labels=None):
        tvs = self.model.encoder.trainable_variables \
                + self.model.decoder.trainable_variables

        with tf.GradientTape() as tape:
            tape.watch(tvs)
            loss, nll, adv = self.model.adversarial_loss(inputs, labels=labels)

        grad = tape.gradient(loss, tvs)
        self.genopt.apply_gradients(zip(grad, tvs))
        return loss, nll, adv

    def update_discriminator(self, codes, prior,
                             label=None,
                             plabel=None):
        tvs = self.model.critic.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(tvs)
            loss = self.model.critic_loss(codes, prior,
                                          label, plabel)
        grad = tape.gradient(loss, tvs)
        self.crtopt.apply_gradients(zip(grad, tvs))
        return loss

    def training_step(self, inputs, prior,
                      label=None, plabel=None):

        tf.keras.backend.set_learning_phase(True)
        for _ in range(self.niters_critic):
            codes = self.model.encode(inputs).sample()
            critic_loss = self.update_discriminator(codes, prior, label, plabel)
        _, recon_loss, generator_loss = self.update_adversarial(inputs)

        tf.compat.v1.assign(self.gs, self.gs + 1)

        return recon_loss, generator_loss, critic_loss
