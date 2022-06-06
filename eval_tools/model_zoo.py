import os
import sys
import logging
import time
import shutil
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from typing import Sequence
from pprint import pformat
from tqdm import tqdm
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

from collections import namedtuple
from eval_tools.eval_tool import evaluate_feature, dropout, imputation_error, plot_image, plot_imputed_heatmap
from eval_tools.utils import add_handler
from eval_tools.dataset import SingleCellDatasetDRA


Results = namedtuple("Results", "X X_rec feature label predict_label cell_type extra_label idx")


class EvalProtocol():
    def __init__(self, hparams, model_name, need_imputation=False):
        self.images = {}
        self.metrics = {"seed":hparams.seed}
        self.hparams = hparams
        self.dataset = SingleCellDatasetDRA(**vars(hparams))
        self.origin_adata = self.dataset.adata.copy()
        self.saved_adata = self.origin_adata.copy()
        self.hparams.in_dim = self.dataset.shape[1]
        self.dataset_name = hparams.name
        self.imputation_drop_rate = hparams.imputation_drop_rate
        self.need_imputation = need_imputation
        self.model_name = model_name
        self.log = logging.getLogger(self.model_name)
        self.log.setLevel(logging.INFO)
        save_dir = os.path.join(hparams.comment, f"{model_name}", hparams.name)
        time_stamp = add_handler(self.log, save_dir, time_stamp=hparams.time_stamp)
        save_dir = os.path.join(save_dir, time_stamp)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.log.info(f"args: {' '.join(sys.argv)}")
        self.register_value("dataset", self.dataset.name)
        self.register_value("cell_num", self.dataset.shape[0])
        self.register_value("gene_num", self.dataset.shape[1])
        self.register_value("save_dir", self.save_dir)
        self.register_value("comment", hparams.comment)
        try:
            self.register_value("latent_dim", hparams.latent_dim)
            if not isinstance(hparams.hidden_dims, Sequence):
                hparams.hidden_dims = [hparams.hidden_dims]
            self.register_value("hidden_dims", "-".join(map(str, hparams.hidden_dims)))
        except:
            pass


    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser.add_argument("--comment", default="de-bug", type=str)
        parent_parser.add_argument("--seed", default=666, type=int)
        parent_parser.add_argument("--imputation_drop_rate", default=0.1, type=int)
        parent_parser.add_argument("--time_stamp", default=None, type=str)

        command_group = parent_parser.add_mutually_exclusive_group()
        command_group.add_argument("--report", action="store_true")
        command_group.add_argument("--reports", default=None, nargs="+", type=str)
        command_group.add_argument("--reports_plot", default=None, nargs="+", type=str)
        command_group.add_argument("--h5_file", default=None, type=str)

        parent_parser.add_argument("--dpi", default=150, type=int)
        parent_parser.add_argument("--max_point_num", default=10000, type=int)
        parent_parser = SingleCellDatasetDRA.add_argparse_args(parent_parser)
        return parent_parser

    def before_imputation_analyse(self, drop_rate=0.1):
        adata = self.origin_adata.copy()
        X_zero, i, j, ix = dropout(adata.layers["count"], rate=drop_rate)
        adata.uns["i"] = i
        adata.uns["j"] = j
        adata.uns["ix"] = ix
        adata.X = X_zero
        return adata

    def on_imputation_analyse(self, zero_adata):
        """X have filled zero
            return X_imputation
        """
        raise NotImplementedError

    def imputation_analyse(self, imputation_drop_rate=0.1):
        zero_adata = self.before_imputation_analyse(drop_rate=imputation_drop_rate)
        X_zero = zero_adata.X.copy()
        X_imputation = self.on_imputation_analyse(zero_adata)
        counts = self.origin_adata.layers["count"]
        i, j, ix = zero_adata.uns["i"], zero_adata.uns["j"], zero_adata.uns["ix"]
        self.saved_adata.obsm["X_random_rec"] = X_imputation
        self.saved_adata.uns["i"], self.saved_adata.uns["j"], self.saved_adata.uns["ix"] = i, j, ix
        self.saved_adata.uns["drop_rate"] = imputation_drop_rate
        ans = imputation_error(X_imputation, counts, X_zero, i, j, ix)
        img = plot_imputed_heatmap(X_imputation, counts, i=i, j=j, ix=ix)
        self.register_image(f"random{imputation_drop_rate}_imputed", img)
        for key, value in ans.items():
            self.register_value(key, value)
        self.log.info(pformat(ans))
        return ans

    def fit_transform(self, adata):
        raise NotImplementedError

    def feature_analyse(self, result):
        """feature label predict_label"""
        save_file = os.path.join(self.save_dir, "feature_plot.h5")
        ans, images, ann_labels = evaluate_feature(feature=result.feature,
                                                   label=result.label,
                                                   predict_label=result.predict_label,
                                                   cell_type=result.cell_type,
                                                   comment="all",
                                                   dpi=self.hparams.dpi,
                                                   logger=self.log,
                                                   max_point_num=self.hparams.max_point_num,
                                                   extra_label=result.extra_label,
                                                   save_file=save_file,
                                                   idx=result.idx)
        self.log.info(pformat(ans))
        for key, value in images.items():
            self.register_image(key, value)
        for key, value in ans.items():
            self.register_value(key, value)
        return ann_labels

    def denoised_pca_analyse(self, X, X_rec, labels, idx=None):
        if X is None or X_rec is None:
            return
        X_adata = ad.AnnData(X)
        sc.pp.normalize_per_cell(X_adata, min_counts=0)
        sc.pp.log1p(X_adata)

        X_rec_adata = ad.AnnData(X_rec)
        sc.pp.normalize_per_cell(X_rec_adata, min_counts=0)
        sc.pp.log1p(X_rec_adata)

        errors = imputation_error(X_mean=X_rec, X=X)
        errors = {f"denoised_{key}":value for key, value in errors.items()}
        for key, value in errors.items():
            self.register_value(key, value)
        self.log.info(pformat(errors))
        rec_images = plot_image(latent=X_rec_adata.X, labels=labels, show=False,
                                frameon=True, dpi=self.hparams.dpi, idx=idx,
                                max_point_num=self.hparams.max_point_num,
                                save_file=os.path.join(self.save_dir, "rec_plot.h5"))
        for key, value in rec_images.items():
            self.register_image(f"rec_{key}", value)

    def run(self):
        start_time = time.time()
        if self.hparams.report:
            self.collect_result(self.hparams.comment)
        elif self.hparams.reports:
            self.collect_results(self.hparams.reports)
        elif self.hparams.reports_plot:
            self.collect_results(self.hparams.reports_plot, plot=True)
        elif self.hparams.h5_file is not None:
            assert os.path.exists(self.hparams.h5_file)
            self.evaluate_from(self.hparams.h5_file)
        else:
            adata = self.origin_adata.copy()
            res = self.fit_transform(adata)
            if res.feature is not None:
                self.saved_adata.obsm["feature"] = res.feature
            if res.X_rec is not None:
                self.saved_adata.obsm["X_rec"] = res.X_rec
            if res.predict_label is not None:
                self.saved_adata.obsm["predict_label"] = res.predict_label
            self.save(self.save_dir)
            if res.feature is not None:
                ann_labels = self.feature_analyse(res)
            else:
                ann_labels = adata.obs["cell_type"]
            if self.need_imputation:
                try:
                    self.log.info("denoised pca analyse ...")
                    self.denoised_pca_analyse(X=res.X, X_rec=res.X_rec, labels=ann_labels, idx=res.idx)
                    self.log.info(f"random dropout {self.imputation_drop_rate} imputation analyse ...")
                    self.imputation_analyse(self.imputation_drop_rate)
                except Exception:
                    self.log.info(traceback.print_exc())
            self.save(self.save_dir)
        self.log.info(f"time cost {time.time()-start_time}")
        self.register_value("cost_time", time.time()-start_time)

    def save(self, save_dir):
        for key, image in self.images.items():
            plt.imsave(os.path.join(save_dir, f'{self.model_name}_{self.dataset_name}_{key}.jpg'), image)
        save_file = os.path.join(save_dir, f"{self.model_name}_{self.dataset_name}.csv")
        if len(self.metrics):
            pd.DataFrame(self.metrics, index=[0]).to_csv(save_file, index=False, sep="\t")
        self.saved_adata.write_h5ad(os.path.join(save_dir, "result.h5"))

    def register_image(self, key, image):
        self.images[key] = image

    def register_value(self, key, value):
        self.metrics[key] = value

    @classmethod
    def collect_results(cls, save_dirs, plot=False):
        ans = []
        for save_dir in save_dirs:
            ans.append(cls.collect_result(save_dir))
            # if plot:
            #     cls.plot(save_dir)
        ans = pd.concat(ans)
        save_file = os.path.join("exp_results", f"{'_'.join(save_dirs)}_report.xlsx")
        ans.to_excel(save_file, engine="openpyxl", index=False)

    @staticmethod
    def valid_paths(save_dir):
        for model in os.listdir(save_dir):
            model_dir = os.path.join(save_dir, model)
            for dataset in tqdm(os.listdir(model_dir)):
                dataset_dir = os.path.join(model_dir, dataset)
                files = [file for file in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, file))]
                for file in files:
                    log_dir = os.path.join(dataset_dir, file)
                    yield log_dir, model, dataset, file

    @classmethod
    def load_result_csv(cls, log_dir):
        target_file = [f for f in os.listdir(log_dir) if f.endswith(".csv")]
        if len(target_file):
            target_file = target_file[0]
            data = pd.read_csv(os.path.join(log_dir, target_file), sep="\t", dtype=str)
            return data


    @classmethod
    def plot(cls, save_dir):
        import matplotlib.pyplot as plt
        from torch.utils.tensorboard import SummaryWriter
        for model in os.listdir(save_dir):
            model_dir = os.path.join(save_dir, model)
            # if model not in ["SAVER"]:
            #     continue
            for dataset in tqdm(os.listdir(model_dir)):
                dataset_dir = os.path.join(model_dir, dataset)
                files = [file for file in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, file))]
                for file in files:
                    log_dir = os.path.join(dataset_dir, file)
                    target_files = [f for f in os.listdir(log_dir) if f.endswith(".jpg")]
                    img_dir = os.path.join("image_log", save_dir, dataset, model, file)
                    if len(target_files):
                        if os.path.exists(img_dir):
                            shutil.rmtree(img_dir)
                        with SummaryWriter(log_dir=img_dir) as writer:
                            for target_file in target_files:
                                image_file = os.path.join(log_dir, target_file)
                                img = plt.imread(image_file)
                                tag = "_".join(target_file.split("_")[-2:]).split(".")[0]
                                writer.add_image(tag=tag, img_tensor=img, global_step=0, dataformats="HWC")

    def evaluate_from(self, h5_file):
        self.log.info(f"load h5 file from {h5_file}")
        adata = sc.read_h5ad(h5_file)
        self.saved_adata = adata.copy()
        start_time = time.time()
        extra_label = None
        if hasattr(adata.obs, "extra_label"):
            extra_label = adata.obs["extra_label"]
            assert len(extra_label) == len(adata)
        extra_label = self.dataset.adata.obs["extra_label"]
        assert len(extra_label)==len(adata)
        res = Results(feature=adata.obsm["feature"],
                      label=adata.obs["label"],
                      predict_label=adata.obsm.get("predict_label"),
                      cell_type=adata.obs["cell_type"],
                      X_rec=adata.obsm.get("X_rec"),
                      X=adata.layers["count"],
                      extra_label=extra_label,
                      idx=adata.obs["idx"])
        ann_labels = self.feature_analyse(res)
        if self.need_imputation:
            try:
                self.log.info("denoised pca analyse ...")
                self.denoised_pca_analyse(X=res.X, X_rec=res.X_rec, labels=ann_labels)

                X_imputation = self.saved_adata.obsm["X_random_rec"]
                i, j, ix = self.saved_adata.uns["i"], self.saved_adata.uns["j"], self.saved_adata.uns["ix"]
                imputation_drop_rate = self.saved_adata.uns["drop_rate"]
                counts = self.saved_adata.layers["count"]
                img = plot_imputed_heatmap(X_imputation, counts, i=i, j=j, ix=ix)
                self.register_image(f"random{imputation_drop_rate}_imputed", img)
                self.log.info(f"random dropout {imputation_drop_rate} imputation analyse ...")
                ans = imputation_error(X_imputation, counts, None, i, j, ix)
                for key, value in ans.items():
                    self.register_value(key, value)
                self.log.info(pformat(ans))
            except Exception:
                self.log.info(traceback.print_exc())
        self.save(self.save_dir)
        self.log.info(f"time cost {time.time()-start_time}")
        self.register_value("cost_time", time.time()-start_time)




try:
    from models.model import autoCell
    import pytorch_lightning as pl
    import torch
    from models.utils import get_loader, Message
except:
    pass


class ModelEvaluator(EvalProtocol):
    def __init__(self, hparams):
        hparams.preprocess = "scvi"
        super(ModelEvaluator, self).__init__(hparams,
                                             model_name=hparams.model.upper(),
                                             need_imputation=True)
        pl.seed_everything(hparams.seed)
        hparams.n_centroid = self.dataset.n_centroid if hparams.n_centroid is None else hparams.n_centroid
        train_dataset, val_dataset = self.dataset.split_train_and_test(hparams.split_rate)
        self.loader = get_loader(self.dataset, hparams, shuffle=False)
        self.train_loader = get_loader(train_dataset, hparams, shuffle=hparams.shuffle)
        self.val_loader = get_loader(val_dataset, hparams, shuffle=False)
        self.model_cls = globals()[hparams.model]
        self.register_value("epochs", hparams.max_epochs)
        self.register_value("sample", hparams.sample)
        self.register_value("give_mean", hparams.give_mean)
        self.register_value("batch_size", hparams.batch_size)
        self.register_value("n_centroid", hparams.n_centroid)
        self.register_value("lr", hparams.lr)
        self.register_value("kl_weight", hparams.kl_weight)
        self.register_value("kl_frozen_rate", hparams.kl_frozen_rate)
        self.register_value("min_kl_weight", hparams.min_kl_weight)
        try:
            self.register_value("neighbor_num", hparams.neighbor_num)
        except:
            pass
        pl.seed_everything(self.hparams.seed)
        self.model = self.model_cls(self.hparams, **vars(self.hparams))

    def init_message_queue(self, input_queue=None, output_queue=None):
        self.model.init_message_queue(input_queue=input_queue, output_queue=output_queue)

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser.add_argument('--model', type=str, default='autoCell', choices=["autoCell"])
        parent_parser.add_argument("--log_dir", type=str, default="runs")
        parent_parser.add_argument("--batch_size", type=int, default=128)
        parent_parser.add_argument("--num_workers", type=int, default=0)
        parent_parser.add_argument("--sample", action="store_true")
        parent_parser.add_argument("--give_mean", action="store_true")
        parent_parser.add_argument("--n_centroid", type=int, default=None)
        parent_parser.set_defaults(latent_dim=10)
        parent_parser.set_defaults(hidden_dims=(128, 128))
        parent_parser = EvalProtocol.add_argparse_args(parent_parser)
        parent_parser = pl.Trainer.add_argparse_args(parent_parser)
        parent_parser = autoCell.add_model_specific_args(parent_parser)
        return parent_parser


    def fit_transform(self, adata):
        max_epochs = self.hparams.max_epochs
        model = self.model
        stage_num = model.max_stage_num
        self.hparams.log_num = self.hparams.log_num // stage_num
        self.hparams.gpus = 1 if torch.cuda.is_available() else 0
        start_epoch = 0
        global_step = 0
        for stage in range(stage_num):
            end_epoch = min(start_epoch+max_epochs//stage_num, max_epochs)
            lr_callback = pl.callbacks.LearningRateMonitor("epoch")
            model.training_stage = stage
            trainer = pl.Trainer.from_argparse_args(self.hparams,
                                                    callbacks=[lr_callback],
                                                    max_epochs=end_epoch,
                                                    terminate_on_nan=False,
                                                    gradient_clip_val=0.01,
                                                    default_root_dir=self.save_dir,
                                                    num_sanity_val_steps=0)
            trainer.current_epoch = start_epoch
            trainer.global_step = global_step
            trainer.fit(model=model, train_dataloader=self.train_loader,
                        val_dataloaders=self.val_loader)
            global_step = trainer.global_step+1
            start_epoch = end_epoch
        trainer.test(model, [self.val_loader, self.train_loader])
        for key, value in model.results.items():
            self.register_value(key, value)
        self.log.info(pformat(model.results))
        dataset = SingleCellDatasetDRA(name=self.dataset.name,
                                       adata=adata)
        dataloader = get_loader(dataset, self.hparams, shuffle=False)
        outputs = trainer.predict(model, dataloader)
        feature = np.concatenate([batch["feature"] for batch in outputs])
        X_rec = np.concatenate([batch["X_rec"] for batch in outputs])

        # trainer.test(model, dataloader)

        predict_label = None
        if hasattr(model, "predict_label"):
            predict_prob = model.predict_label(feature).cpu().numpy()
            predict_label = np.argmax(predict_prob, axis=-1)
            self.saved_adata.obsm["X_prob"] = predict_prob
        extra_label = None
        if hasattr(adata.obs, "extra_label"):
            extra_label = adata.obs["extra_label"]
        res = Results(feature=feature,
                      X=adata.layers["count"],
                      X_rec=X_rec,
                      label=adata.obs["label"],
                      predict_label=predict_label,
                      cell_type=adata.obs["cell_type"],
                      extra_label=extra_label,
                      idx=adata.obs["idx"])
        self.trainer = trainer
        return res

    def on_imputation_analyse(self, zero_adata):
        sc.pp.log1p(zero_adata)
        dataset = SingleCellDatasetDRA(name=self.dataset.name,
                                       adata=zero_adata)
        dataloader = get_loader(dataset, self.hparams, shuffle=False)
        feature = self.model.get_denoised_samples(dataloader, sample=self.hparams.sample)
        return feature

