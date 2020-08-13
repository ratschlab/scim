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

The PROSSTT simulated dataset and the metastatic melanoma dataset can be downloaded from the TuPro website: https://tpreports.nexus.ethz.ch/downloads/scim/

Details to access the human bone marrow dataset can be found in the publication that describes it, Oetjen et al., 2018 (https://insight.jci.org/articles/view/124928)

### Simulated data generated with PROSTT

Using PROSSTT (Papadopouloset al., 2019), we generate three single-cell ’omics-styled technologies which share a common latent structure without direct feature correspondences. PROSSTT parameterizes a negative binomial distribution given a tree representing an underlying temporal branching process. By using the same tree and running PROSSTT under different seeds, we obtain three datasets with a common latent structure yet lacking any correspondences between features. We used a five branch tree with different branch lengths. Each dataset contains 64,000 cells with 256 markers.


### Single-cell profile of a metastatic melanoma sample from the Tumor Profiler Consortium

**CyTOF data:** The sample was profiled with CyTOF using a 41-markers panel designed for an in-depth characterization of the immune compartmentof a sample. Data preprocessing was performed following the workflow described in (Chevrier et al., 2017, 2018). Cell-type assignment was performed using a Random Forest classifier trained on multiple manually gated samples. In the SCIM manuscript, we utilize a subset comprising B-Cells and T-Cells only, for a total of 135,334 cells.

**scRNA-seq data:** In brief, standard QC-measures and preprocessing steps, such as removal of low quality cells, as well as filtering out mitochondrial, ribosomal and non-coding genes, were applied to 10X Genomics-generated data. Expression data was library-size normalized and corrected for the cell-cycle effect. Cell-type identification was performed using a set of cell-type-specific marker genes (Tirosh et al., 2016). Genes were then filtered to retain those that could code for proteins measured in CyTOF channels, the top 32 T-Cell/ B-Cell marker genes, and the remaining most variable genes for a final set of 256 genes. The total number of B-Cells and T-Cells in this dataset amounts to 4,683. Only T cells and B cells are provided.

More details can be found in the SCIM manuscript (https://www.biorxiv.org/content/10.1101/2020.06.11.146845v3)
