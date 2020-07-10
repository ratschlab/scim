import anndata
import tensorflow as tf
import pandas as pd

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


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


def plot_training(trainer, axes):
    assert len(axes) >= 3
    loss_keys = trainer.history['train'].keys()

    mse_keys = [k for k in loss_keys if k.startswith('mse-')]
    disc_keys = [k for k in loss_keys if k.startswith('discriminator-')]
    probs_keys = [k for k in loss_keys if k.startswith('probs-')]

    plot_history_keys(trainer, mse_keys + disc_keys, axes[0])
    plot_history_keys(trainer, probs_keys, axes[1])
    plot_history_keys(trainer, ['divergence'], axes[2])
    return


def plot_history_keys(trainer, keys, ax):
    tab10 = plt.get_cmap('tab10')
    handles = list()
    for idx, key in enumerate(keys):
        trail = trainer.history['train'].get(key)

        if trail is not None:
            step, vals = list(zip(*trail))
            ax.plot(step, vals, label=key, alpha=0.5, color=tab10(idx))

        trail = trainer.history['test'].get(key)

        step, vals = list(zip(*trail))
        ax.plot(step, vals, label=key, alpha=0.75, color=tab10(idx))

        handles.append(Line2D([0], [0], color=tab10(idx), label=key))
    ax.legend(handles=handles)

    return

def adata_to_pd(adata, add_cell_code_name=None):
    ''' A helper function to load anndata and convert to pandas
    '''
    pd_data = pd.DataFrame(adata.X)
    if(add_cell_code_name is not None):
        pd_data.index = [add_cell_code_name+'_'+str(x) for x in pd_data.index.array]
    return pd_data
