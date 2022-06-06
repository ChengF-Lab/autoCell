# Graph Embedded Gaussian Mixture Variational Autoencoder Network for End-to-End Analysis of Single-Cell/Nucleus RNA-Sequencing Data

autoCell is a variational autoencoding network that combines graph embedding and probabilistic depth Gaussian mixture model to infer the distribution of high-dimensional sparse scRNA-seq data.


With autoCell, you can:
* Build a low-dimensional representation of the single-cell gene expression data.
* Visualize the cell clustering results and the gene expression patterns.
* Remove the dropout effect by taking the count structure, overdispersed nature and sparsity of the data into account using zero-inflated negative binomial (ZINB) loss function.

## Dependencies
The code has been tested with the following versions of packages.
* Python 3.6 (we recommend [Anaconda](https://www.continuum.io/downloads) distribution)
* pytorch=1.8.1
* pytorch-lightning=1.2.10
* anndata=0.7.6
* scanpy=1.7.2

## Dataset
The path for the dataset could be./autoCell/Zeisel/<dataset_name>  
For example, the Zeisel dataset could be in the folder./ autoCell/Zeisel/Zeisel as follows:  
The Zeisel dataset consists of 3,005 cells from the mouse brain.  
In addition, the Zeisel dataset has the ground truth labels of 7 distinct cell types.  
There were 2000 highest variance genes selected in the Zeisel dataset.  

## Usage
```
git clone https://github.com/ChengF-Lab/autoCell.git
cd autoCell
python debug_main.py --help
```

## Demo
`bash zeisel_exp.sh`
