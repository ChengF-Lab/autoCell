import os
import time
import logging


def add_handler(log, log_dir, time_stamp=None):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    format = '%Y-%m-%d %H-%M-%S'
    if time_stamp is None:
        time_stamp = time.strftime(format, time.localtime())
    fm = logging.Formatter(fmt="[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
                           datefmt="%m/%d %H:%M:%S")
    file_handler = logging.FileHandler(os.path.join(log_dir,
                                                    f"{time_stamp}.log"))
    file_handler.setFormatter(fm)
    log.addHandler(file_handler)
    log.info(log_dir)
    return time_stamp

def get_loader(dataset, hparams, shuffle=False):
    from torch.utils.data import DataLoader, sampler
    if shuffle:
        random_sampler = sampler.RandomSampler(dataset)
        batch_sampler = sampler.BatchSampler(random_sampler, batch_size=hparams.batch_size, drop_last=False)
        train_loader = DataLoader(dataset, sampler=batch_sampler, pin_memory=True, batch_size=None,
                                  num_workers=hparams.num_workers)
        return train_loader
    else:
        sequential_sampler = sampler.SequentialSampler(dataset)
        test_sampler = sampler.BatchSampler(sequential_sampler, batch_size=hparams.batch_size, drop_last=False)
        test_loader = DataLoader(dataset, sampler=test_sampler, pin_memory=True, batch_size=None,
                                 num_workers=hparams.num_workers)
        return test_loader

