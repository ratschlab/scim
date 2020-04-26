# scim
Code for Universal Single-Cell Matching with Unpaired Feature Sets

Integrates datasets from multiple single cell 'omics technologies in two steps:
* Constructs a technology invariant latent space
* Matches cells across technologies by bipartite matching of latent representations
  
integration.py
  Main models for producing the integrated latent space
 
matching.py
  Functions to perform bipartite matching
  
trainer.py
  Class to train integration model
  
utils
  activations.py
    Wraps network outputs in tfp distributions
  
  discriminator.py
    Discriminator classes including Spectral Normalization mixin
   
  evaluate.py
    Latent space evaluation (e.g. divergence score)
