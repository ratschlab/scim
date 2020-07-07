import tensorflow as tf


class CriticBase(tf.keras.Model):
    def __init__(self,
                 net,
                 input_cats=None):

        super(CriticBase, self).__init__()
        self.net = net
        self.input_cats = input_cats
        self.can_condition = self.input_cats is not None
        return

    def loss_real(self, logits):
        raise NotImplementedError

    def loss_fake(self, logits):
        raise NotImplementedError

    def condition_inputs(self, data, label_indices):
        '''Append one-hot labels to input data'''
        assert self.can_condition
        n_classes = self.input_cats.size

        # tf.one_hot is VERY picky about how indices are passed
        if isinstance(label_indices, tf.Tensor):
            indices = label_indices.numpy().tolist()
        else:
            indices = list(label_indices)

        encoded_labels = tf.one_hot(indices, n_classes, dtype=data.dtype)
        return tf.concat((data, encoded_labels), axis=1)

    def logits(self, data, labels=None):
        if self.can_condition:
            assert labels is not None
            data = self.condition_inputs(data, labels)
        return self.net(data)

    def loss(self, data, real, labels=None):
        '''Discriminator loss

        Try to classify real data as true, generated data as fake.
        '''
        logits = self.logits(data, labels)

        if real:
            loss = self.loss_real(logits)
        else:
            loss = self.loss_fake(logits)

        return loss


class NonSaturatingCritic(CriticBase):
    def __init__(self, *args, **kwargs):
        super(NonSaturatingCritic, self).__init__(*args, **kwargs)
        return

    def loss_fake(self, logits):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits,
                labels=tf.zeros_like(logits)
               )

        return loss

    def loss_real(self, logits):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits,
                labels=tf.ones_like(logits)
               )

        return loss


class SpectralNormMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.u_store = dict()
        for layer in self.net.layers:
            if not hasattr(layer, 'kernel'):
                continue

            w = layer.kernel
            self.u_store[w.name] = tf.random.normal(
                    (w.shape[0], 1),
                    dtype=w.dtype)

        return

    def _l2norm(self, arg, eps=1e-12):
        return arg / (tf.norm(arg) + eps)

    def power_iterate(self, w):
        assert w.name in self.u_store
        u = self.u_store.get(w.name)

        v = self._l2norm(tf.matmul(w, u, transpose_a=True))
        u = self._l2norm(tf.matmul(w, v))
        sigma = tf.matmul(u, tf.matmul(w, v), transpose_a=True)
        self.u_store[w.name] = u
        return sigma

    def normalize(self, modify=True):
        sigma_lut = dict()
        for layer in self.net.layers:
            if not hasattr(layer, 'kernel'):
                continue

            w = layer.kernel
            name = w.name
            sigma = self.power_iterate(w)

            if modify:
                tf.compat.v1.assign(w, w/sigma)

            sigma_np = sigma.numpy()
            assert sigma_np.size == 1
            sigma_np = sigma_np[0, 0]

            sigma_lut[name] = sigma_np

        return sigma_lut

    def loss(self, *args, normalize=None, **kwargs):

        if normalize is None:
            normalize = tf.keras.backend.learning_phase()

        if normalize:
            self.normalize()

        return super().loss(*args, **kwargs)


class SpectralNormCritic(SpectralNormMixin, NonSaturatingCritic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
