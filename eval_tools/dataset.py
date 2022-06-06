#!/usr/bin/env python
"""
#
#

# File Name: dataset.py
# Description:

"""
import time
import os
import numpy as np
import pandas as pd
import scipy
from scipy.io import mmread

import scanpy as sc
import anndata as ad

class SingleCellDatasetDRA():
    """
    "10x_68k", "10x_73k", "Macosko", "Zeisel", "data1", "yan"
    """
    def __init__(self, name="Zeisel",
                 # root_dir="/home/xjl/PycharmProject/DRA",
                 root_dir="/home/jm/PycharmProjects/lcc/jupyter notebook workspace/DRA",
                 transpose=False,
                 preprocess="scvi",
                 adata=None,
                 n_top_genes=2000,
                 gene_min_counts=3,
                 **kwargs):
        assert preprocess in ["scvi", "dca", None]
        extra_label_files = {"dataA2":"dataF2",
                             "dataF2":"dataA2",
                             "kolo":"kolo1",
                             "kolo1":"kolo"}
        h5files = ["Zeisel", "oligo", "mg", "astro"]
        if preprocess=="scvi":
            self.preprocess = self.scvi_preprocess
        elif preprocess=="dca":
            self.preprocess = self.dca_preprocess
        self.name = name
        self.gene_num = None
        self.cell_num = None
        if adata is None:
            if name in h5files:
                self.adata = sc.read_h5ad(os.path.join(root_dir, name, f"{name}.h5"))
            else:
                data, cell_type, gene = self.load_data(root_dir, name, transpose=transpose)
                if name in extra_label_files:
                    _, extra_label, _ = self.load_data(root_dir, extra_label_files[name], transpose=transpose)
                    assert len(extra_label)==len(cell_type)
                idx = np.array(data.index) if isinstance(data, pd.DataFrame) else np.arange(data.shape[0])
                data = np.array(data)
                data[data < 0] = 0.0
                cell_type_set = sorted(set(cell_type))
                label2id = {label: idx for idx, label in enumerate(cell_type_set)}
                label = pd.Categorical([label2id[label] for label in cell_type], ordered=True)
                self.adata = ad.AnnData(data, obs={"cell_type": cell_type, "label":label, "idx":idx},
                                        var={"gene":gene},
                                        uns={"n_centroid":len(cell_type_set)})
                if name in extra_label_files:
                    self.adata.obs["extra_label"] = extra_label
    #         for transform in transforms:
    #             self.data = transform(self.data)
    #         self.data = self.data / np.max(self.data)
            self.cell_num, self.gene_num = self.adata.shape
            sc.pp.filter_genes(self.adata, min_counts=gene_min_counts)
            sc.pp.highly_variable_genes(
                    self.adata,
                    n_top_genes=min(n_top_genes, self.adata.shape[1]),
                    subset=True,
                    # layer="counts",
                    flavor="seurat_v3")
            self.adata.layers["count"] = self.adata.X.copy()
            self.adata.raw = self.adata
            self.preprocess()
        else:
            self.adata = adata
        self.data = {key:value.to_numpy() for key, value in self.adata.obs.items() if key!="cell_type"}
        self.data["x"] = self.adata.X
        self.data["count"] = self.adata.layers["count"]
        self.n_centroid = int(self.adata.uns["n_centroid"])
        print(self.adata)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("dataset")
        parser.add_argument('--name', type=str, default="10x_73k",
                            choices=["10x_68k", "10x_73k", "Zeisel",
                                     "GSE138852", "kolo", "romov", "sim05", "sim25", "Klein",
                                     "petro"])
        parser.add_argument('--root_dir', type=str, default=".")
        parser.add_argument('--preprocess', type=str, default="scvi", choices=["scvi", "dca"])
        parser.add_argument("--split_rate", type=float, default=0.9)
        parser.add_argument("--gene_min_counts", type=int, default=3)
        parser.add_argument("--n_top_genes", type=int, default=2000)
        parser.add_argument("--shuffle", action="store_true")
        return parent_parser

    def load_data(self, root_dir, name, transpose=False):
        mtx_files = ["10x_68k", "10x_73k", "Zeisel"]
        txt_files = [ "GSE138852", "kolo", "romov", "sim05", "sim25", "Klein",
                     ]
        print(f"Loading data '{name}' from {os.path.join(root_dir, name)} ...")
        t0 = time.time()
        if name in mtx_files:
            data, label, gene = self.load_mtx_data(root_dir, name)
        elif name in txt_files:
            data, label, gene = self.load_txt_data(root_dir, name)
        else:
            raise NotImplementedError
        if transpose:
            data = data.transpose()
        assert data.shape[0] == label.shape[0]
        print('Original data contains {} cells x {} peaks'.format(*data.shape))
        print("Finished loading takes {:.2f} min".format((time.time() - t0) / 60))
        label = pd.Categorical(label)
        return data, label, gene

    def load_mtx_data(self, root_dir, name):
        data = mmread(os.path.join(root_dir, name, "sub_set-720.mtx")).toarray().astype("float32")
        label = np.loadtxt(os.path.join(root_dir, name, "labels.txt")).astype("int").astype("str")
        return data, label, None

    def load_txt_data(self, root_dir, name):
        data = pd.read_csv(os.path.join(root_dir, name, f"{name}.txt"), sep="\t", index_col=0).transpose()
        label = pd.read_csv(os.path.join(root_dir, name, f"{name}_label.txt"), sep="\t", index_col=0, names=["label"])
        # data.index = data.index.astype(label.index.dtype)
        if not np.all(data.index == label.index):
            print("数据与标签无法对齐！")
        return data, label["label"].values.astype("str"), data.columns.to_list()

    def __len__(self):
        return self.adata.shape[0]
        # return 10

    def __getitem__(self, index):
        keys = ["x", "label", "count"]
        return {key:self.data[key][index] for key in keys}

    @property
    def shape(self):
        return self.adata.shape

    @property
    def label(self):
        return self.adata.obs["label"].values

    @property
    def cell_type(self):
        return self.adata.obs["cell_type"].values

    def preprocess(self):
        pass

    def split_train_and_test(self, split_rate=0.9):
        index = np.random.permutation(self.adata.shape[0])
        train_index = index[:int(len(index)*split_rate)]
        test_index = index[int(len(index)*split_rate):]
        train_data = self.adata[train_index].copy()
        test_data = self.adata[test_index].copy()
        return SingleCellDatasetDRA(name=self.name, adata=train_data), SingleCellDatasetDRA(name=self.name, adata=test_data)

    def compute_local_library(self, adata, batch_key="batch"):
        if batch_key not in adata.obs_keys():
            adata.obs[batch_key] = np.ones(adata.shape[0], dtype=int)
        local_means = np.zeros((adata.shape[0], 1))
        local_vars = np.zeros((adata.shape[0], 1))
        batch_indices = adata.obs[batch_key]
        for i_batch in np.unique(batch_indices):
            idx_batch = np.squeeze(batch_indices == i_batch)
            data = adata[idx_batch].layers["count"]
            sum_counts = data.sum(axis=1)
            masked_log_sum = np.ma.log(sum_counts)
            assert not np.ma.is_masked(masked_log_sum), "This dataset has some empty cells,\
                    this might fail inference. Data should be filtered with `scanpy.pp.filter_cells()`"
            log_counts = masked_log_sum.filled(0)
            local_mean = (np.mean(log_counts).reshape(-1, 1)).astype(np.float32)
            local_var = (np.var(log_counts).reshape(-1, 1)).astype(np.float32)
            local_means[idx_batch], local_vars[idx_batch] = local_mean, local_var
        return local_means, local_vars

    def scvi_preprocess(self, adata=None, log_variational=True):
        adata = self.adata if adata is None else adata
        if log_variational:
            sc.pp.log1p(adata)
        local_mean, local_var = self.compute_local_library(adata)
        adata.obs["local_library_mean"] = local_mean
        adata.obs["local_library_var"] = local_var

    def dca_preprocess(self, adata=None, size_factors=True, logtrans_input=True, normalize_input=True):
        adata = self.adata if adata is None else adata
        if size_factors:
            sc.pp.normalize_per_cell(adata)
            adata.obs['library_size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
        else:
            adata.obs['library_size_factors'] = 1.0
        if logtrans_input:
            sc.pp.log1p(adata)
        if normalize_input:
            sc.pp.scale(adata)
