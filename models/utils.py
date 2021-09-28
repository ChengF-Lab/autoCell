import os
import time
import logging
from tqdm import tqdm
import torch

from eval_tools.eval_tool import evaluate


def add_handler(log, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    format = '%Y-%m-%d %H-%M-%S'
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


class MessageManager():
    def __init__(self, writer=None, input_queue=None, output_queue=None):
        self.writer = writer
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.job_num = 0

    def init_message_queue(self, input_queue=None, output_queue=None):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.job_num = 0

    def set_writer(self, writer):
        self.writer = writer

    def log_fn(self, ans):
        for key, value in ans.items():
            self.writer.add_scalar(f"training/{key}", value, global_step=ans["global_step"])

    def send(self, message):
        if self.output_queue is None:
            ans = message.process()
            self.log_fn(ans)
        else:
            self.job_num += 1
            self.output_queue.put(message)

    def read(self, block=False):
        if self.input_queue is not None:
            while block or not self.input_queue.empty():
                ans = self.input_queue.get()
                self.job_num -= 1
                self.log_fn(ans)
                if block:
                    break

    def read_all(self):
        if self.input_queue is not None:
            for _ in tqdm(range(self.job_num)):
                ans = self.input_queue.get()
                self.log_fn(ans)
        self.job_num = 0


class Message():
    def __init__(self, feature=None, label=None, extra_info=None, global_step=None, end=False):
        self.extra_info = {} if extra_info is None else extra_info
        self.feature = feature
        self.label = label
        self.global_step = global_step
        self.end = end

    def process(self):
        if self.is_end():
            return {}
        feature, label = self.feature, self.label
        if isinstance(feature, torch.Tensor):
            feature = feature.numpy()
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        info = evaluate(feature, label, name="kmeans")
        info["global_step"] = self.global_step
        info.update(self.extra_info)
        return info

    def is_end(self):
        return self.end
