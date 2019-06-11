from torch import manual_seed
import torch.utils.data

def CreateDataset(opt):
    from data.cloudnet_dataset import CloudNetDataset
    dataset = CloudNetDataset()
    print("Dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader():
    def __init__(self, opt):
        self.opt = opt
        self.dataset = CreateDataset(opt)

        def init_fn(worker_id):
            manual_seed(opt.seed)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=opt.nThreads,
            worker_init_fn=init_fn,
        )

    def name(self):
        return 'CustomDatasetDataLoader'

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
