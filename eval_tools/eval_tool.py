import logging
import traceback
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score,\
    adjusted_rand_score, homogeneity_score,\
    completeness_score, silhouette_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from pprint import pformat
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages

def evaluate(latent, labels, n_centroid=None, name="kmeans", return_label=False, log=None):
    if log is None:
        log = logging.getLogger("lightning")
    if n_centroid is None:
        n_centroid = len(np.unique(labels))
    try:
        cl, score = clustering(latent, k=n_centroid, name=name)
        eval_scores = measure(cl, labels)
        eval_scores["score"] = score
        eval_scores["n_centroid"] = n_centroid
        eval_scores = {f"{name[0]}_{key}": value for key, value in eval_scores.items()}
        if return_label:
            log.info(f"{name}_distribution:{Counter(cl)}")
    except Exception:
        eval_scores = {}
        cl = np.zeros_like(labels)
        log.info(f"evaluate {name} error !")
        log.info(traceback.print_exc())
    if return_label:
        return eval_scores, cl
    return eval_scores

def plot2np(fig):
    import io
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr


def plot_image(latent, labels, idx=None, show=True, frameon=False, max_point_num=-1, save_file=None, **kwargs):
    max_point_num = min(len(latent), max_point_num)
    if max_point_num<0:
        max_point_num = len(latent)
    mask = np.random.permutation(len(latent))[:max_point_num]
    plot_feature = latent[mask]
    idx = np.array(idx.values)[mask]
    if isinstance(labels, pd.Series):
        plot_image_label = np.array(labels.values)[mask]
    elif isinstance(labels, pd.DataFrame):
        plot_image_label = labels.iloc[mask]
    else:
        plot_image_label = labels[mask]
    adata = ad.AnnData(plot_feature, obs=plot_image_label)
    # tsne_img = plot_tSNE(latent=plot_feature, labels=plot_image_label, show=show, frameon=frameon, adata=adata, **kwargs)
    umap_img = plot_umap(latent=plot_feature, labels=plot_image_label, show=show, frameon=frameon, adata=adata, **kwargs)
    # pca_img = plot_pca(latent=plot_feature, labels=plot_image_label, show=show, frameon=frameon, adata=adata, **kwargs)
    adata.obs["idx"] = idx
    if save_file:
        adata.write_h5ad(save_file)
    return {"umap":umap_img}


from functools import wraps
def dpi_keeper(fn):
    @wraps(fn)
    def wraper(*args, **kwargs):
        old_dpi = rcParams["figure.dpi"]
        rcParams["figure.dpi"] = kwargs.pop("dpi", 80)
        ans = fn(*args, **kwargs)
        rcParams["figure.dpi"] = old_dpi
        return ans
    return wraper

@dpi_keeper
def plot_heatmap(x, y, label=None, show=False, pdf=False,**kwargs):
    data = {"x": x.reshape(-1),
            "y": y.reshape(-1)}
    hue = None
    if label is not None:
        data["label"] = np.array(label).reshape(-1)
        hue = "label"
    data = pd.DataFrame(data)
    fig = plt.figure()
    sns.scatterplot(data=data, x="x", y="y", hue=hue, s=1)
    sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako")
    max_loc = max(max(x), max(y))
    min_loc = min(min(x), min(y))
    sns.lineplot([min_loc, max_loc], [min_loc, max_loc], color="black")
    ax = plt.gca()
    ax.set_xlabel("log10(Imputed+1)")
    ax.set_ylabel("log10(Real+1)")
    plt.tight_layout()
    if not show:
        fig = plt.gcf()
        img = plot2np(fig)
        plt.close()
        return img
    else:
        plt.show()

@dpi_keeper
def plot_tSNE(latent, labels, show=True, frameon=False, adata=None, **kwargs):
    if adata is None:
        if len(labels.shape)==1:
            adata = ad.AnnData(latent, obs={"label":labels})
        else:
            adata = ad.AnnData(latent, obs=labels)
    sc.tl.tsne(adata)
    sc.pl.tsne(adata, color=adata.obs.columns, frameon=frameon, show=show, **kwargs)
    plt.tight_layout()
    if not show:
        fig = plt.gcf()
        img = plot2np(fig)
        return img

@dpi_keeper
def plot_umap(latent, labels, show=True, frameon=False, adata=None, **kwargs):
    if adata is None:
        if len(labels.shape) == 1:
            adata = ad.AnnData(latent, obs={"label": labels})
        else:
            adata = ad.AnnData(latent, obs=labels)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata, min_dist=0.2)
    sc.pl.umap(adata, color=adata.obs.columns, frameon=frameon, show=show, **kwargs)
    plt.tight_layout()
    if not show:
        fig = plt.gcf()
        img = plot2np(fig)
        return img

@dpi_keeper
def plot_pca(latent, labels, show=True, frameon=False, adata=None, **kwargs):
    if adata is None:
        if len(labels.shape) == 1:
            adata = ad.AnnData(latent, obs={"label": labels})
        else:
            adata = ad.AnnData(latent, obs=labels)
    sc.tl.pca(adata)
    sc.pl.pca(adata, color=adata.obs.columns, frameon=frameon, show=show, **kwargs)
    plt.tight_layout()
    if not show:
        fig = plt.gcf()
        img = plot2np(fig)
        return img

def plot_imputed_heatmap(X_mean, X, i=None, j=None, ix=None, show=False, dpi=150):
    if i is None or j is None or ix is None:
        x = np.reshape(X_mean, -1)
        y = np.reshape(X, -1)
    else:
        all_index = i[ix], j[ix]
        x, y = X_mean[all_index], X[all_index]
    x = np.log10(x+1)
    y = np.log10(y+1)
    img = plot_heatmap(x=x, y=y, show=show, dpi=dpi)
    return img

def plot_imputed_heatmap_pdf(X_mean, X, i=None, j=None, ix=None, show=False, dpi=150, save_file=None):
    if i is None or j is None or ix is None:
        x = np.reshape(X_mean, -1)
        y = np.reshape(X, -1)
    else:
        all_index = i[ix], j[ix]
        x, y = X_mean[all_index], X[all_index]
    x = np.log10(x+1)
    y = np.log10(y+1)
    data = {"x": x.reshape(-1),
            "y": y.reshape(-1)}
    hue = None
    if y is not None:
        data["label"] = np.array(y).reshape(-1)
        hue = "label"
    data = pd.DataFrame(data)
    pdf = PdfPages(save_file)
    fig = plt.figure()
    sns.scatterplot(data=data, x="x", y="y", hue=hue, s=1)
    sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako")
    max_loc = max(max(x), max(y))
    min_loc = min(min(x), min(y))
    sns.lineplot([min_loc, max_loc], [min_loc, max_loc], color="black")
    ax = plt.gca()
    ax.set_xlabel("log10(Imputed+1)")
    ax.set_ylabel("log10(Real+1)")
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    pdf.close()

def measure(predicted, true):
    NMI = normalized_mutual_info_score(true, predicted)
    # print("NMI:" + str(NMI))
    RAND = adjusted_rand_score(true, predicted)
    # print("RAND:" + str(RAND))
    HOMO = homogeneity_score(true, predicted)
    # print("HOMOGENEITY:" + str(HOMO))
    COMPLETENESS = completeness_score(true, predicted)
    # print("COMPLETENESS:" + str(COMPLETENESS))
    ACC = cluster_acc(y_pred=predicted, y_true=true)
    return {'NMI': NMI, 'RAND': RAND, 'HOMOGENEITY': HOMO, 'COMPLETENESS': COMPLETENESS, "ACC":ACC}

def clustering(points, k=2, name='kmeans'):
    '''
    points: N_samples * N_features
    k: number of clusters
    '''
    if name == 'kmeans':
        kmeans = KMeans(n_clusters=k, n_init=100).fit(points)
        ## print within_variance
        # cluster_distance = kmeans.transform( points )
        # within_variance = sum( np.min(cluster_distance,axis=1) ) / float( points.shape[0] )
        # print("AvgWithinSS:"+str(within_variance))
        if len(np.unique(kmeans.labels_)) > 1:
            si = silhouette_score(points, kmeans.labels_)
            # print("Silhouette:"+str(si))
        else:
            si = 0
            print("Silhouette:" + str(si))
        return kmeans.labels_, si

    if name == 'spec':
        spec = SpectralClustering(n_clusters=k, affinity='cosine').fit(points)
        si = silhouette_score(points, spec.labels_)
        # print("Silhouette:" + str(si))
        return spec.labels_, si

    if name == "gmm":
        gmm = GaussianMixture(n_components=k, random_state=0, covariance_type="diag")
        label = gmm.fit_predict(points)
        si = silhouette_score(points, label)
        return label, si


def topk_accuracy(output, target, topk=(1, 5)):
    maxk = min(max(topk), output.shape[0])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res[f"top-{k}"] = correct_k / batch_size
    return res


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size



def evaluate_feature(feature, label, predict_label=None, cell_type=None, idx=None, comment="",
                     show_umap=True, show_tsne=True, show_pca=True, dpi=300,
                     writer=None, logger=None, global_step=0, max_point_num=-1, extra_label=None, save_file=None):
    stage = comment
    max_point_num = min(len(feature), max_point_num)
    if max_point_num<0:
        max_point_num = len(feature)
    mask = np.random.permutation(len(feature))[:max_point_num]

    if logger:
        logger.info(f"{stage} true_distribution:{Counter(label)}")
    try:
        methods = ["kmeans"]
        image_label = {"label": pd.Categorical(label)}
        if cell_type is not None:
            image_label["cell_type"] = cell_type
        if extra_label is not None:
            image_label["extra_label"] = pd.Categorical(extra_label)
        res = {}
        images = {}
        for method in methods:
            ans, ans_label = evaluate(feature, label, name=method, return_label=True, log=logger)
            res.update(ans)
            # if len(ans):
            #     image_label[f"{method}_label"] = pd.Categorical(ans_label)
        res = {f"{stage}/{key}": value for key, value in res.items()}

        if predict_label is not None:
            # image_label["predict_label"] = pd.Categorical(predict_label)
            if logger:
                logger.info(f"{stage} predict_distribution:{Counter(predict_label)}")
            accuracy = cluster_acc(y_true=label, y_pred=predict_label)
            res[f"{stage}/predict_ACC"] = accuracy

        image_label = pd.DataFrame(image_label)
        plot_feature = feature[mask]
        plot_image_label = image_label.iloc[mask]
        idx = np.array(idx)[mask]
        adata = ad.AnnData(plot_feature, obs=plot_image_label)
        if show_umap:
            umap_img = plot_umap(latent=plot_feature, labels=plot_image_label, show=False, dpi=dpi, adata=adata)
            images[f"{stage}_umap"] = umap_img

        adata.obs["idx"] = idx
        if save_file:
            adata.write_h5ad(save_file)

        if logger is not None:
            logger.info(f"{pformat(res)}")
        if writer is not None:
            for key, image in images.items():
                writer.add_image(key, image, global_step=global_step,
                                 dataformats="HWC")
            for key, value in res.items():
                writer.add_scalar(key, value, global_step=global_step)
    except:
        logger.info(f"evalueate {stage} error!")
    return res, images, image_label


def dropout(X, rate=0.1):
    """
    X: original testing set
    ========
    returns:
    X_zero: copy of X with zeros
    i, j, ix: indices of where dropout is applied
    """
    X_zero = np.copy(X)
    # select non-zero subset
    i, j = np.nonzero(X_zero)

    # choice number 1 : select 10 percent of the non zero values (so that distributions overlap enough)
    ix = np.random.choice(range(len(i)), int(np.floor(0.1 * len(i))), replace=False)
    X_zero[i[ix], j[ix]] *= np.random.binomial(1, rate)

    # choice number 2, focus on a few but corrupt binomially
    # ix = np.random.choice(range(len(i)), int(slice_prop * np.floor(len(i))), replace=False)
    # X_zero[i[ix], j[ix]] = np.random.binomial(X_zero[i[ix], j[ix]].astype(np.int), rate)
    return X_zero, i, j, ix

# IMPUTATION METRICS
def imputation_error(X_mean, X, X_zero=None, i=None, j=None, ix=None):
    """
    X_mean: imputed dataset
    X: original dataset
    X_zero: zeros dataset
    i, j, ix: indices of where dropout was applied
    ========
    returns:
    median L1 distance between datasets at indices given
    """
    if i is None or j is None or ix is None:
        x = np.reshape(X_mean, -1)
        y = np.reshape(X, -1)
    else:
        all_index = i[ix], j[ix]
        x, y = X_mean[all_index], X[all_index]

    # L1-distance
    l1_distance = np.median(np.abs(x - y))

    # Cosine similarity
    cosine_similarity = np.sum(x*y)/(np.linalg.norm(x)*np.linalg.norm(y)+1e-10)

    # RMSE
    rmse = np.sqrt(mean_squared_error(x, y))

    return {"imputation_l1":l1_distance,
            "imputation_cosine":cosine_similarity,
            "imputation_rmse":rmse}