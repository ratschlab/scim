import pandas as pd
import anndata


def load_anndata(source_path, target_path):
    """Read data from anndata files and convert to desired pandas format
    """
    source = anndata.read(source_path)
    target = anndata.read(target_path)
        
    return source, target


def ann_to_pandas(input_data, add_cell_code_name=None):
    """Convert anndata into pandas dataframe
    inputs:
        input_data: anndata object with latent representation stored in .obs['code'] (cells x features)
        add_cell_code_name: string to create cell codes, when no cell barcodes are present (add_cell_code_name+idx)
    outputs:
        pd_data: pandas dataframe (cells x features)
    """
    pd_data = pd.DataFrame(input_data.X)
        
    if(add_cell_code_name is not None):
        pd_data.index = [add_cell_code_name+'_'+str(x) for x in pd_data.index.array]
        
    return pd_data

