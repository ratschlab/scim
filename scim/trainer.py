import tensorflow as tf
from anndata import AnnData
from scim.evaluate import score_divergence


def switch_obsm(adata, obsm_key, X_name='X'):
    new = AnnData(X=adata.obsm[obsm_key],
                  obs=adata.obs,
                  obsm=adata.obsm,
                  uns=adata.uns)
    new.obsm[X_name] = adata.X
    return new


class Trainer:
    def __init__(self, vae_lut, discriminator,
                 source_key,
                 disopt, genopt_lut,
                 beta):

        self.techs = list(vae_lut.keys())

        self.vae_lut = vae_lut
        self.discriminator = discriminator
        self.disopt = disopt
        self.genopt_lut = genopt_lut
        self.source_key = source_key
        self.niters_discriminator = 5
        self.beta = beta

        self.history = dict()
        self.saver = self._init_saver()
        return

    def _init_saver(self):
        ckpt_lut = dict()
        ckpt_lut.update({f'model-{k}': v for k, v in self.vae_lut.items()})
        ckpt_lut.update({f'opt-{k}': v for k, v in self.genopt_lut.items()})
        ckpt_lut.update({
            'opt-discriminator': self.disopt,
            'discriminator': self.discriminator
            })

        ckpt_lut.update({
            'source_key': tf.Variable(self.source_key),
            'beta': tf.Variable(self.beta),
            'niters_discriminator': tf.Variable(self.niters_discriminator)
            })

        return tf.compat.v1.train.Checkpoint(**ckpt_lut)

    def restore(self, ckpt_dir):
        path = tf.compat.v1.train.latest_checkpoint(ckpt_dir)
        res = self.saver.restore(path)
        res.expect_partial()
        res.assert_nontrivial_match()
        return

    def vae_step(self, tech, inputs, beta=None):
        tf.keras.backend.set_learning_phase(True)
        vae = self.vae_lut[tech]
        opt = self.genopt_lut[tech]

        tvs = vae.trainable_variables
        with tf.GradientTape() as tape:
            loss, (mse, kl), (codes, recon) = vae.call(inputs, beta=beta)

        grads = tape.gradient(loss, tvs)
        opt.apply_gradients(zip(grads, tvs))
        return loss, (mse, kl), (codes, recon)

    def discriminator_loss(self, tech,
                           codes, prior_codes,
                           code_label, prior_label):

        is_source = tech == self.source_key
        data_loss = self.discriminator.loss(codes,
                                            labels=code_label,
                                            real=is_source)

        prior_loss = self.discriminator.loss(prior_codes,
                                             labels=prior_label,
                                             real=not is_source)
        disc_loss = data_loss + prior_loss
        return disc_loss, (data_loss, prior_loss)

    def discriminator_step(self,
                           dtech, ptech,
                           data, prior,
                           dlabel=None, plabel=None,
                           ):

        tf.keras.backend.set_learning_phase(True)
        _, dcodes, _ = self.vae_lut[dtech].forward(data)
        _, pcodes, _ = self.vae_lut[ptech].forward(prior)

        tvs = self.discriminator.trainable_variables
        with tf.GradientTape() as tape:
            disc_loss, (data_loss, prior_loss) = self.discriminator_loss(
                    dtech,
                    dcodes, pcodes,
                    dlabel, plabel)

        grads = tape.gradient(disc_loss, tvs)
        self.disopt.apply_gradients(zip(grads, tvs))
        return disc_loss, (data_loss, prior_loss)

    def adversarial_step(self, tech, data, label=None):

        tf.keras.backend.set_learning_phase(True)
        vae = self.vae_lut[tech]
        opt = self.genopt_lut[tech]
        is_source = tech is self.source_key
        tvs = vae.trainable_variables

        with tf.GradientTape() as tape:
            _, (nll, _), (codes, recon) = vae.call(data)
            adv = self.discriminator.loss(codes,
                                          labels=label,
                                          real=not is_source)
            loss = nll + self.beta * adv

        grads = tape.gradient(loss, tvs)
        opt.apply_gradients(zip(grads, tvs))
        return loss, (nll, adv), (codes, recon)

    def forward(self, tech, adata, labels):
        '''Store model outputs in an anndata structure

        adata: AnnData instance
        tech: key corresponding to adata
        labels: key used to supervise discriminator
                (will use adata.obs[labels])

        Writes:
            embeddings (code) and reconstruction (recon) to adata.obsm
            loss, discriminator logits to adata.obs

        '''
        vae = self.vae_lut[tech]
        _, (mse, kl), (code, recon) = vae.call(adata.X)
        label_values = adata.obs[labels].cat.codes.values
        logits = self.discriminator.logits(code, label_values)
        disc_loss = self.discriminator.loss(logits,
                                            real=tech is self.source_key,
                                            gave_logits=True)

        adata.obsm['code'] = code.numpy()
        adata.obsm['recon'] = recon.numpy()

        adata.obs['loss-mse'] = mse.numpy().ravel()
        adata.obs['loss-kl'] = kl.numpy().ravel()
        adata.obs['loss-discriminator'] = disc_loss.numpy().ravel()
        adata.obs['probs-discriminator'] = tf.sigmoid(logits).numpy().ravel()

        return

    def evaluate(self, test_lut, labels=None, train_phase=False):
        tf.keras.backend.set_learning_phase(train_phase)

        codes = dict()
        for tech, data in test_lut.items():
            self.forward(tech, data, labels)
            codes[tech] = switch_obsm(data, 'code')

        tmp = codes.pop(self.source_key)
        stack = tmp.concatenate(
                codes.values(),
                batch_categories=[self.source_key] + list(codes.keys()),
                batch_key='tech')

        index = stack.obs.groupby('tech', group_keys=False).apply(lambda x: x.sample(100)).index
        substack = stack[index]
        divergence = score_divergence(substack.X, substack.obs[labels].cat.codes, substack.obs['tech'].cat.codes)
        stack.uns['divergence'] = divergence
        return stack

    def record(self, key, lut, step):
        hist = self.history.setdefault(key, dict())
        for k, v in lut.items():
            trail = hist.setdefault(k, list())
            trail.append((step, v))
        return
