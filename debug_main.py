from eval_tools.model_zoo import ModelEvaluator
import warnings
warnings.filterwarnings("ignore")
from torch.multiprocessing import Process, Queue
from models.utils import Message

def log_process_fn(in_queue, out_queue):
    message = in_queue.get()
    res = []
    while not message.is_end():
        ans = message.process()
        res.append(ans)
        out_queue.put(ans)
        message = in_queue.get()


if __name__=="__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser = ModelEvaluator.add_argparse_args(parser)
    hparams = parser.parse_args()
    # hparams.hidden_dims = []
    in_queue = Queue()
    out_queue = Queue()
    log_process = Process(target=log_process_fn, args=(out_queue, in_queue))
    log_process.start()
    evaluator = ModelEvaluator(hparams)
    evaluator.init_message_queue(input_queue=in_queue, output_queue=out_queue)
    try:
        evaluator.run()
    finally:
        evaluator.model.message_manager.send(Message(end=True))
    log_process.join()


    # evaluator = ModelEvaluator(hparams)
    # evaluator.run()
    # from eval_tools.models import ZIFAEvaluator
    # evaluator = ZIFAEvaluator(hparams)
    # evaluator.run()