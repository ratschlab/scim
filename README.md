# scim
Code for Universal Single-Cell Matching with Unpaired Feature Sets

Integrates datasets from multiple single cell 'omics technologies in two steps:
* Constructs a technology invariant latent space
* Matches cells across technologies by bipartite matching of latent representations
  
See demo.ipynb for how to setup the framework, train the model and peform the matches.

To replicate the environment run:
```
conda create --name scim python==3.6.9 pip
conda activate scim
pip install -r requirements.txt
```
## Datasets used

The PROSSTT simulated dataset and the metastatic melanoma dataset can be
downloaded from the TuPro website: https://tpreports.nexus.ethz.ch/downloads/scim/

Details to access the human bone marrow dataset can be found in the publication that
describes it, Oetgen et al., 2018 (https://insight.jci.org/articles/view/124928)
