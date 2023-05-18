from queue import Queue
import torch.utils.data
from data.base_data_loader import BaseDataLoader
from threading import Thread


def CreateDataset(opt):
    train_dataset = None
    if opt.phase == 'train':
        from data.audio_dataset import AudioDataset
        train_dataset = AudioDataset(opt)
        eval_dataset = AudioDataset(opt, True)
        print("dataset [%s] was created" % (train_dataset.name()))
        print("dataset [%s] was created" % (eval_dataset.name()))
    elif opt.phase == 'test':
        from data.audio_dataset import AudioTestDataset
        train_dataset = AudioTestDataset(opt)
        print("dataset [%s] was created" % (train_dataset.name()))
        eval_dataset = None

    # dataset.initialize(opt)
    return train_dataset, eval_dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        self.q_size = 16
        self.idx = 0
        self.load_stream = torch.cuda.Stream(device='cuda')
        self.queue: Queue = Queue(maxsize=self.q_size)
        BaseDataLoader.initialize(self, opt)
        self.train_dataset, self.eval_dataset = CreateDataset(opt)
        self.data_lenth = len(self.train_dataset)
        self.eval_data_lenth = len(self.eval_dataset) if self.eval_dataset is not None else None
        if opt.phase == "train":
            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=opt.batchSize,
                shuffle=True,
                num_workers=int(opt.nThreads),
                prefetch_factor=8,
                pin_memory=True)

            self.eval_dataloder = torch.utils.data.DataLoader(
                self.eval_dataset,
                batch_size=opt.batchSize,
                shuffle=True,
                num_workers=int(opt.nThreads),
                pin_memory=True)

        elif opt.phase == "test":
            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=opt.batchSize,
                num_workers=int(opt.nThreads),
                shuffle=False,
                pin_memory=True)
            self.eval_dataloder = None
            self.eval_data_lenth = 0
    
    def load_loop(self) -> None:  # The loop that will load into the queue in the background
        for i, sample in enumerate(self.train_dataloader):
            self.queue.put(self.load_instance(sample))
            if i == len(self):
                break
    
    def load_instance(self, sample:dict):
        with torch.cuda.stream(self.load_stream):
            return {k:v.cuda(non_blocking=True) for k,v in sample.items()}

    def get_train_dataloader(self):
        return self.train_dataloader
    
    def async_load_data(self):
        return self

    def get_eval_dataloader(self):
        return self.eval_dataloder

    def eval_data_len(self):
        return self.eval_data_lenth

    def __len__(self):
        return self.data_lenth
    
    def __iter__(self):
        if_worker = not hasattr(self, "worker") or not self.worker.is_alive()  # type: ignore[has-type]
        if if_worker and self.queue.empty() and self.idx == 0:
            self.worker = Thread(target=self.load_loop)
            self.worker.daemon = True
            self.worker.start()
        return self

    def __next__(self):
        # If we've reached the number of batches to return
        # or the queue is empty and the worker is dead then exit
        done = not self.worker.is_alive() and self.queue.empty()
        done = done or self.idx >= len(self)
        if done:
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        # Otherwise return the next batch
        out = self.queue.get()
        self.queue.task_done()
        self.idx += 1
        return out

