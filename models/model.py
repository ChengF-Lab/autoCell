import os
import torch
from torch import optim
import pytorch_lightning as pl
from tqdm import tqdm
from pprint import pformat
import pandas as pd
import traceback
from functools import wraps

from pytorch_lightning import _logger as log

from eval_tools.eval_tool import evaluate_feature, cluster_acc
from eval_tools.dataset import SingleCellDatasetDRA
from models.utils import MessageManager, Message, add_handler, get_loader
from models.model_helper import *

def enabled_only(fn):
    @wraps(fn)
    def wrapped_fn(self, *args, **kwargs):
        if self.enabled:
            fn(self, *args, **kwargs)
    return wrapped_fn

class LogModel(pl.LightningModule):
    def __init__(self, hparams, input_queue=None, output_queue=None, **kwargs):
        super(LogModel, self).__init__()
        # self.hparams = hparams
        self.message_manager = MessageManager(input_queue=input_queue, output_queue=output_queue)
        self.results = {}
        self.images = {}
        self.log_interval = None
        self.stage = ["test", "train", "all"]
        self.training_stage = 0
        self.max_stage_num = 1
        self.save_hyperparameters(hparams)

    @property
    def enabled(self):
        return self.trainer.current_epoch % self.log_interval == 0 if self.log_interval else False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser.add_argument("--lr", type=float, default=1e-3)
        parent_parser.add_argument("--min_lr", type=float, default=1e-6)
        parent_parser.add_argument("--patience", default=10, type=int)
        parser = parent_parser.add_argument_group("logger")
        parser.add_argument('--sample_num', type=int, default=5000, help="训练中测试时采样样本数")
        parser.add_argument('--log_num', type=int, default=100, help="训练中测试的次数")
        parser.add_argument('--no_show_umap', action='store_true')
        parser.add_argument("--no_show_tsne", action='store_true')
        return parent_parser

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def on_train_start(self):
        if self.hparams.log_num:
            self.log_interval = max(self.trainer.max_epochs // self.hparams.log_num, 2)
        else:
            self.log_interval = self.trainer.max_epochs
        self.message_manager.set_writer(self.logger.experiment)
        if self.training_stage == 0:
            self.save_dir = self.trainer.checkpoint_callback.dirpath
            self.time_stamp = add_handler(log, self.save_dir)

    def loss_fn(self, batch, batch_idx=None, optimizer_idx=None):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        info = self.loss_fn(batch, batch_idx)
        if self.enabled:
            info["label"] = batch["label"]
            return info
        if torch.any(torch.isnan(info["loss"])):
            self.trainer.should_stop = True
        return info["loss"]

    @enabled_only
    def training_epoch_end(self, outputs):
        current_epoch = self.current_epoch
        loss_keys = [key for key in outputs[0] if key.startswith("loss")]
        loss_info = {key: sum([batch[key] for batch in outputs]).item() / len(outputs) for key in loss_keys}
        features = torch.cat([batch["feature"] for batch in outputs]).detach().cpu()
        labels = torch.cat([batch["label"] for batch in outputs]).cpu()
        mask = torch.randperm(features.shape[0])[:self.hparams.sample_num]
        feature = features[mask]
        label = labels[mask]
        message = Message(feature=feature, label=label,
                          global_step=current_epoch, extra_info=loss_info)
        self.train_loss = loss_info
        self.message_manager.send(message)
        self.message_manager.read()

    def validation_step(self, batch, batch_idx=None, dataloader_idx=None):
        info = self.loss_fn(batch)
        return info

    def validation_epoch_end(self, outputs):
        loss_keys = [key for key in outputs[0] if key.startswith("loss")]
        loss_info = {key: sum([batch[key] for batch in outputs]).item() / len(outputs) for key in loss_keys}
        self.log("valing/loss", loss_info["loss"], on_epoch=True)
        writer = self.logger.experiment
        for key, value in loss_info.items():
            writer.add_scalar(f"val/{key}", value, global_step=self.current_epoch)

    def test_step(self, batch, batch_idx=None, dataloader_idx=None):
        info = self.loss_fn(batch, batch_idx)
        info["label"] = batch["label"]
        return info

    def test_epoch_end(self, outputs):
        if len(self.trainer.num_test_batches)==1:
            outputs = [outputs]
        results, images = {}, {}
        for dataloader_idx, output in enumerate(outputs):
            stage = self.stage[dataloader_idx]
            feature = torch.cat([batch["feature"] for batch in output]).detach().cpu().numpy()
            label = torch.cat([batch["label"] for batch in output]).cpu().numpy()
            predict_label = None
            log.info(f"{stage} true_distribution:{Counter(label)}")
            writer = self.trainer.logger.experiment
            try:
                if hasattr(self, "predict_label"):
                    predict_score = self.predict_label(torch.from_numpy(feature).to(self.device))
                    predict_label = torch.argmax(predict_score, dim=-1).cpu().numpy()

                res, images, image_label = evaluate_feature(feature=feature, label=label,
                                                            predict_label=predict_label,
                                                            comment=stage,
                                                            logger=log, writer=writer,
                                                            global_step=self.trainer.max_epochs,
                                                            dpi=80)

                results.update(res)
                images.update(images)
            except Exception:
                log.info(traceback.print_exc())
        log.info(f"{pformat(results)}")
        self.results = results
        self.images = images
        save_file = os.path.join(self.save_dir, f"{self.time_stamp}.csv")
        pd.DataFrame(results, index=[0]).to_csv(save_file, index=False, sep="\t")

    def init_message_queue(self, input_queue=None, output_queue=None):
        self.message_manager.init_message_queue(input_queue=input_queue, output_queue=output_queue)

    def on_train_end(self):
        log.info("wait for exit ...")
        # self.message_manager.send(Message(end=True))
        self.message_manager.read_all()


class ClusterModel(LogModel):
    def __init__(self, hparams, **kwargs):
        super(ClusterModel, self).__init__(hparams, **kwargs)
        self.dataset = None

    def get_batch_latent_representation(self, batch, give_mean=True):
        raise NotImplementedError

    @torch.no_grad()
    def get_latent_representation(self, data_loader, give_mean=True):
        state = self.training
        self.eval()
        features = []
        for batch in tqdm(data_loader):
            batch = {key: value.to(self.device) for key, value in batch.items()}
            feature = self.get_batch_latent_representation(batch, give_mean=give_mean)
            features.append(feature)
        features = torch.cat(features, dim=0).cpu().numpy()
        self.training = state
        self.train(self.training)
        return features


    def get_batch_denoised_samples(self, batch, sample=False):
        raise NotImplementedError

    @torch.no_grad()
    def get_denoised_samples(self, data_loader, sample=False):
        state = self.training
        self.eval()
        features = []
        for batch in tqdm(data_loader):
            batch = {key: value.to(self.device) for key, value in batch.items()}
            output = self.get_batch_denoised_samples(batch, sample=sample)
            features.append(output)
        features = torch.cat(features, dim=0).cpu().numpy()
        self.training = state
        self.train(state)
        return features

    def configure_optimizers(self):
        optimizer = optim.Adam(lr=self.hparams.lr, params=self.parameters(), weight_decay=1e-5)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True,
                                                            patience=self.hparams.patience,
                                                            min_lr=self.hparams.min_lr)
        return [optimizer], [{"scheduler":lr_scheduler,
                              "monitor":"valing/loss",
                              "interval":"epoch"}]

    def __build_dataset(self):
        if self.dataset is None:
            self.dataset = SingleCellDatasetDRA(**vars(self.hparams.hparams))
            train_dataset, test_dataset = self.dataset.split_train_and_test(self.hparams.split_rate)
            self.train_loader = get_loader(train_dataset, self.hparams, shuffle=self.hparams.shuffle)
            self.val_loader = get_loader(test_dataset, self.hparams, shuffle=False)
            self.test_loader = get_loader(test_dataset, self.hparams, shuffle=False)

    def train_dataloader(self):
        self.__build_dataset()
        return self.train_loader

    def val_dataloader(self):
        self.__build_dataset()
        return self.val_loader

    def test_dataloader(self):
        self.__build_dataset()
        return self.test_loader


class AE(ClusterModel):
    def __init__(self, hparams, in_dim, latent_dim, hidden_dims=tuple(),
                 distribution="gaussian", **kwargs):
        super(AE, self).__init__(hparams, **kwargs)
        self.dims = [in_dim] + list(hidden_dims)+[latent_dim]
        self.encoder = Block(in_dim, latent_dim, hidden_dims)
        self.decoder = Block(in_dim, latent_dim, hidden_dims, reverse=True, dropout=0)
        self.register_buffer("eps", torch.tensor(1e-10))
        self.register_buffer("one", torch.tensor(1.0))
        self.reconstruction_loss_fn = ReconstructionLoss(distribution)

    @staticmethod
    def add_model_specific_args(parent_parser):
        ClusterModel.add_model_specific_args(parent_parser)
        parent_parser.add_argument('--in_dim', type=int)
        parent_parser.add_argument('--latent_dim', type=int, default=10)
        parent_parser.add_argument('--hidden_dims', type=int, default=(128, 128), nargs="+")
        parent_parser.add_argument("--distribution", type=str, default="gaussian",
                                   choices=["gaussian", "bernoulli", "possion", "nb", "zinb"])
        return parent_parser

    def get_batch_latent_representation(self, batch, give_mean=True):
        x = batch["x"]
        z = self.encoder(x)
        return z

    def get_batch_denoised_samples(self, batch, sample=False):
        x = batch["x"]
        z = self.encoder(x)
        x_bar = self.decoder(z)
        if sample:
            x_bar = self.reconstruction_loss_fn.sample(x_bar)
        return x_bar

    def loss_fn(self, batch, batch_idx=None, optimizer_idx=None):
        x = batch["x"]
        z = self.encoder(x)
        x_bar = self.decoder(z)
        loss = self.reconstruction_loss_fn(x_bar, x).sum(dim=-1).mean(dim=-1)
        return {"loss": loss,
                "loss_rec": loss,
                "feature": z}


class VAE(AE):
    def __init__(self, hparams, in_dim, latent_dim, hidden_dims=tuple(), kl_weight=1.0, **kwargs):
        super(VAE, self).__init__(hparams, in_dim, latent_dim, hidden_dims, **kwargs)
        mean_header = nn.Linear(self.dims[-2], self.dims[-1])
        log_var_header = nn.Linear(self.dims[-2], self.dims[-1])
        self._kl_weight = kl_weight
        self.encoder = Block(in_dim, latent_dim, hidden_dims, headers=[mean_header,
                                                                       log_var_header])
        self.sampler = NormalSampler()
        self.kl_divergence_fn = self.sampler.kl_divergence

    @staticmethod
    def add_model_specific_args(parent_parser):
        AE.add_model_specific_args(parent_parser)
        parent_parser.add_argument('--kl_weight', type=float, default=1.0)
        parent_parser.add_argument('--kl_frozen_rate', default=0.0, type=float)
        parent_parser.add_argument('--min_kl_weight', type=float, default=0.1)
        return parent_parser

    def on_train_start(self):
        super(AE, self).on_train_start()
        max_step = (self.trainer.max_epochs-self.trainer.current_epoch) * len(self.trainer.train_dataloader)
        frozen_step = int(max_step*self.hparams.kl_frozen_rate)
        self.kl_weight_scheduler = WarmUpScheduler(max_step,
                                                   min_value=self.hparams.min_kl_weight,
                                                   max_value=self._kl_weight,
                                                   frozen_step=frozen_step)

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        super(AE, self).on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)
        self.kl_weight_scheduler.step()

    @property
    def kl_weight(self):
        weight = self.kl_weight_scheduler.compute()
        return torch.tensor(weight, device=self.device)

    def get_batch_denoised_samples(self, batch, sample=False):
        x = batch["x"]
        mean, log_var = self.encoder(x)
        z = self.sampler(mean, log_var)
        x_bar = self.decoder(z)
        if sample:
            x_bar = self.reconstruction_loss_fn.sample(x_bar)
        return x_bar

    def get_batch_latent_representation(self, batch, give_mean=True):
        x = batch["x"]
        mean, log_var = self.encoder(x)
        if not give_mean:
            z = self.sampler(mean, log_var)
            return z
        return mean

    def loss_fn(self, batch, batch_idx=None, optimizer_idx=None):
        x = batch["x"]
        mean, log_var = self.encoder(x)
        z = self.sampler(mean, log_var)
        x_bar = self.decoder(z)
        reconstruction_loss = self.reconstruction_loss_fn(x_bar, x).sum(dim=-1).mean(dim=-1)
        kl_loss = self.kl_divergence_fn(mean, log_var, z=z).mean()
        loss = reconstruction_loss + self.kl_weight * kl_loss
        return {"loss": loss,
                "loss_kl": kl_loss,
                "loss_rec": reconstruction_loss,
                "feature": mean,
                "log_var":log_var,
                "mean":mean,
                "z":z}

    @enabled_only
    def training_epoch_end(self, outputs):
        super(VAE, self).training_epoch_end(outputs)
        writer = self.logger.experiment
        mean = torch.cat([batch["mean"] for batch in outputs]).cpu()
        log_var = torch.cat([batch["log_var"] for batch in outputs]).cpu()
        z = torch.cat([batch["z"] for batch in outputs]).cpu()
        writer.add_histogram("z", z, global_step=self.current_epoch)
        writer.add_histogram("log_var", log_var.exp(), global_step=self.current_epoch)
        writer.add_histogram("mean", mean, global_step=self.current_epoch)
        writer.add_scalar("training/kl_weight", self.kl_weight, global_step=self.current_epoch)



class SCVI(VAE):
    def __init__(self, hparams, in_dim, latent_dim, hidden_dims=tuple(),
                 distribution="zinb", **kwargs):
        super(SCVI, self).__init__(hparams, in_dim, latent_dim, hidden_dims,
                                   distribution=distribution, **kwargs)
        scale_header = nn.Linear(self.dims[1], self.dims[0])
        dropout_header = nn.Linear(self.dims[1], self.dims[0])
        self.decoder.headers = nn.ModuleList([scale_header, dropout_header])
        self.px_r = torch.nn.Parameter(torch.randn(in_dim))
        self.kl_divergence_fn = self.sampler.kl_divergence

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = VAE.add_model_specific_args(parent_parser)
        parent_parser.set_defaults(distribution="zinb")
        return parent_parser

    def get_batch_denoised_samples(self, batch, sample=False):
        x, count = batch["x"], batch["count"]
        library = torch.log(count.sum(1)).unsqueeze(1)
        mean, log_var = self.encoder(x)
        z = self.sampler(mean, log_var)
        px_scale, px_dropout = self.decoder(z)
        px_scale = F.softmax(px_scale, dim=-1)
        px_rate = torch.exp(library) * px_scale
        px_r = torch.exp(self.px_r)
        denoise_x = px_rate
        if sample:
            denoise_x = self.reconstruction_loss_fn.sample(px_rate, px_r)
        return denoise_x

    def loss_fn(self, batch, batch_idx=None, optimizer_idx=None):
        x, count = batch["x"], batch["count"]
        library = torch.log(count.sum(1)).unsqueeze(1)
        mean, log_var = self.encoder(x)
        z = self.sampler(mean, log_var)
        px_scale, px_dropout = self.decoder(z)
        px_scale = F.softmax(px_scale, dim=-1)
        px_rate = torch.exp(library) * px_scale
        px_r = torch.exp(self.px_r)
        reconstruction_loss = self.reconstruction_loss_fn(mu=px_rate,
                                                          y=count,
                                                          theta=px_r,
                                                          drop_prob=px_dropout).sum(dim=-1).mean(dim=-1)
        kl_loss = self.kl_divergence_fn(mean, log_var, z=z).mean()

        loss = reconstruction_loss + self.kl_weight * kl_loss
        return {"loss": loss,
                "loss_kl": kl_loss,
                "loss_rec": reconstruction_loss,
                "feature": mean,
                "mean":mean,
                "log_var":log_var,
                "z":z}

    def predict(self, batch, batch_idx=None, dataloader_idx=None, give_mean=True, sample=False):
        x, count = batch["x"], batch["count"]
        library = torch.log(count.sum(1)).unsqueeze(1)
        mean, log_var = self.encoder(x)
        z = self.sampler(mean, log_var)
        px_scale, px_dropout = self.decoder(z)
        px_scale = F.softmax(px_scale, dim=-1)
        px_rate = torch.exp(library) * px_scale
        px_r = torch.exp(self.px_r)
        denoise_x = px_rate
        if sample:
            denoise_x = self.reconstruction_loss_fn.sample(px_rate, px_r)
        feature = mean if give_mean else z
        return {"feature":feature,
                "X_rec":denoise_x}

class VADE(SCVI):
    def __init__(self, hparams, in_dim, latent_dim, hidden_dims=tuple(), n_centroid=1,
                 train_gmm_from="z", **kwargs):
        super(VADE, self).__init__(hparams, in_dim, latent_dim, hidden_dims, **kwargs)
        self.gmm_sampler = GMMSampler(latent_dim=latent_dim, n_centroid=n_centroid, trainable=False)
        self.n_centroid = n_centroid
        self.kl_divergence_fn = self.gmm_sampler.kl_divergence
        self.max_stage_num = 3
        self.train_gmm_from = train_gmm_from
        self.stage_lr = {"stage_0": self.hparams.lr,
                         # "stage_1":self.lr*0.1,
                         "stage_2":self.hparams.lr*0.1
                         }

    def on_train_start(self):
        super(VADE, self).on_train_start()
        if self.training_stage == 1:
            data_loader = self.trainer.train_dataloader
            sample = self.get_latent_representation(data_loader, give_mean=(self.train_gmm_from == "mean"))
            self.gmm_pi, self.gmm_mu, self.gmm_var = self.gmm_sampler.fit(sample)

    @enabled_only
    def training_epoch_end(self, outputs):
        super(VADE, self).training_epoch_end(outputs)
        writer = self.logger.experiment
        label = torch.cat([batch["label"] for batch in outputs]).cpu()
        prob = torch.cat([batch["prob"] for batch in outputs]).cpu()
        try:
            predict_label = torch.argmax(prob, dim=-1).cpu()
            accuracy = cluster_acc(y_true=label.numpy(), y_pred=predict_label.numpy())
            writer.add_scalar("training/predict_ACC", accuracy, global_step=self.current_epoch)
        except:
            pass
        writer = self.logger.experiment
        writer.add_histogram("mu_c", self.gmm_sampler.mu_c, global_step=self.current_epoch)
        writer.add_histogram("var_c", self.gmm_sampler.var_c, global_step=self.current_epoch)
        writer.add_histogram("pi", self.gmm_sampler._pi, global_step=self.current_epoch)

    @enabled_only
    def on_train_epoch_end(self, outputs):
        # optimizer_idx, batch_idx, btpp_idx
        if self.training_stage >= 2:
            key = self.train_gmm_from
            feature = torch.cat([batch[0]["extra"][key] for batch in outputs[0]]).detach().cpu().numpy()
            self.gmm_pi, self.gmm_mu, self.gmm_var = self.gmm_sampler.fit(feature)


    def configure_optimizers(self):
        lr = self.stage_lr.get(f"stage_{self.training_stage}", self.hparams.lr)
        if self.training_stage == 1:
            parameters = [{"params": self.decoder.parameters()},
                          {"params": self.gmm_sampler.parameters()},
                          {"params": self.px_r}]
        else:
            parameters = self.parameters()
        optimizer = optim.Adam(lr=lr, params=parameters, weight_decay=1e-5)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True,
                                                            patience=self.hparams.patience,
                                                            min_lr=self.hparams.min_lr)
        return [optimizer], [{"scheduler":lr_scheduler,
                              "monitor":"valing/loss",
                              "interval":"epoch"}]

    def loss_fn(self, batch, batch_idx=None, optimizer_idx=None):
        x, count = batch["x"], batch["count"]
        library = torch.log(count.sum(1)).unsqueeze(1)
        mean, log_var = self.encoder(x)
        z = self.sampler(mean, log_var)
        px_scale, px_dropout = self.decoder(z)
        px_scale = F.softmax(px_scale, dim=-1)
        px_rate = torch.exp(library) * px_scale
        px_r = torch.exp(self.px_r)
        reconstruction_loss = self.reconstruction_loss_fn(mu=px_rate,
                                                          y=count,
                                                          theta=px_r,
                                                          drop_prob=px_dropout).sum(dim=-1).mean(dim=-1)
        kl_loss, info = self.kl_divergence_fn(mean, log_var, z=z)
        kl_loss = kl_loss.mean()
        loss = reconstruction_loss + self.kl_weight * kl_loss
        info.update({"loss": loss,
                     "loss_kl": kl_loss,
                     "loss_rec": reconstruction_loss,
                     "feature": mean,
                     "z": z,
                     "mean": mean,
                     "log_var": log_var})
        return info

    @torch.no_grad()
    def predict_label(self, feature):
        if isinstance(feature, np.ndarray):
            feature = torch.from_numpy(feature).to(device=self.device)
        return self.gmm_sampler.get_gamma(feature)


    @staticmethod
    def add_model_specific_args(parent_parser):
        SCVI.add_model_specific_args(parent_parser)
        parent_parser.add_argument('--n_centroid', type=int, default=1)
        parent_parser.add_argument('--train_gmm_from', default="z", choices=["mean", "z"], type=str)
        return parent_parser



class VADE2(VADE):
    def __init__(self, hparams, **kwargs):
        super(VADE2, self).__init__(hparams, **kwargs)
        self.gmm_sampler = GMMSampler(latent_dim=kwargs.get("latent_dim"),
                                      n_centroid=kwargs.get("n_centroid"),
                                      trainable=True)
        self.kl_divergence_fn = self.gmm_sampler.kl_divergence

    @enabled_only
    def on_train_epoch_end(self, outputs):
        # optimizer_idx, batch_idx, btpp_idx
        if self.training_stage == 2:
            key = self.train_gmm_from
            feature = torch.cat([batch[0]["extra"][key] for batch in outputs[0]]).detach().cpu().numpy()
            self.gmm_pi, self.gmm_mu, self.gmm_var = self.gmm_sampler.fit(feature)

    def on_train_start(self):
        super(VADE, self).on_train_start()
        if self.training_stage == 1:
            data_loader = self.trainer.train_dataloader
            sample = self.get_latent_representation(data_loader, give_mean=(self.train_gmm_from == "mean"))
            self.gmm_pi, self.gmm_mu, self.gmm_var = self.gmm_sampler.fit(sample)

    def loss_fn(self, batch, batch_idx=None, optimizer_idx=None):
        info = super(VADE2, self).loss_fn(batch, batch_idx, optimizer_idx)
        if self.training_stage>=1:
            loss_pi = F.mse_loss(self.gmm_pi, self.gmm_sampler._pi)
            loss_var = F.mse_loss(self.gmm_var, self.gmm_sampler._var_c)
            loss_mu = F.mse_loss(self.gmm_mu, self.gmm_sampler.mu_c)
            info["loss_pi"] = loss_pi
            info["loss_var"] = loss_var
            info["loss_mu"] = loss_mu
            info["loss"] += loss_pi+loss_mu
        return info


class autoCell(VADE):
    def __init__(self, hparams, in_dim, latent_dim, hidden_dims=tuple(),
                 n_centroid=1, neighbor_num=10, **kwargs):
        super(autoCell, self).__init__(hparams, in_dim, latent_dim, hidden_dims, n_centroid=n_centroid, **kwargs)
        self.dgg_sampler = DGGSampler(latent_dim=latent_dim, n_centroid=n_centroid, neighbor_num=neighbor_num)
        self.max_stage_num = 3
        self.lr = self.hparams.lr
        self.stage_lr["stage_1"] = self.lr*self.hparams.stage_lr_rate[0]
        self.stage_lr["stage_2"] = self.lr*self.hparams.stage_lr_rate[1]
        self.stage_lr["stage_3"] = self.lr*self.hparams.stage_lr_rate[2]

    @staticmethod
    def add_model_specific_args(parent_parser):
        SCVI.add_model_specific_args(parent_parser)
        parent_parser.add_argument('--neighbor_num', type=int, default=10)
        parent_parser.add_argument('--stage_lr_rate', type=float, nargs="+", default=(0.5, 0.125, 0.125))
        return parent_parser

    @torch.no_grad()
    def predict_label(self, feature):
        if isinstance(feature, np.ndarray):
            feature = torch.from_numpy(feature).to(device=self.device)
        return self.dgg_sampler.get_gamma(feature, update_pi=False)

    def on_train_start(self):
        super(autoCell, self).on_train_start()
        if self.training_stage == 2:
            self.kl_divergence_fn = self.dgg_sampler.kl_divergence
            self.dgg_sampler._pi = self.gmm_sampler._pi
            self.dgg_sampler._var_c = self.gmm_sampler._var_c
            self.dgg_sampler.mu_c = self.gmm_sampler.mu_c
            if hasattr(self.gmm_sampler, "gmm"):
                self.dgg_sampler.gmm = self.gmm_sampler.gmm

    @enabled_only
    def training_epoch_end(self, outputs):
        super(autoCell, self).training_epoch_end(outputs)
        if self.training_stage==2:
            sim = torch.cat([batch["sim"] for batch in outputs[:-1]]).cpu()
            writer = self.logger.experiment
            writer.add_histogram("sim", sim, global_step=self.current_epoch)

    @property
    def kl_weight(self):
        weight = self.kl_weight_scheduler.compute()
        if self.training_stage>=2:
            weight = 1.0
        return torch.tensor(weight, device=self.device)