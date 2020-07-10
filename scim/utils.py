import anndata
import tensorflow as tf


def switch_obsm(adata, obsm_key, X_name='X'):
    new = anndata.AnnData(
        X=adata.obsm[obsm_key],
        obs=adata.obs,
        obsm=adata.obsm,
        uns=adata.uns)

    new.obsm[X_name] = adata.X
    return new


def make_network(doutput,
                 units,
                 dinput=None,
                 batch_norm=False,
                 dropout=None,
                 **kwargs):
    if isinstance(units, int):
        units = [units]

    net = list()
    if dinput is not None:
        net.append(tf.keras.layers.InputLayer(dinput))

    for width in units:
        layer = list()
        layer.append(tf.keras.layers.Dense(width))

        if batch_norm:
            layer.append(tf.keras.layers.BatchNormalization())

        layer.append(tf.keras.layers.ReLU())

        if dropout is not None and dropout > 0:
            layer.append(tf.keras.layers.Dropout(dropout))
        net.extend(layer)

    net.append(tf.keras.layers.Dense(doutput))
    return tf.keras.Sequential(net, **kwargs)
