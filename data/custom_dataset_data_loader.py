import torch.utils.data

from data.base_data_loader import BaseDataLoader

def CreateDataset(opt):
    dataset = None
    # if opt.model in ['cloudnet', 'cloudcnn']:
    from data.cloudnet_dataset import CloudNetDataset
    dataset = CloudNetDataset()

    # elif opt.dataset_mode == 'unaligned':
    #     from data.unaligned_posenet_dataset import UnalignedPoseNetDataset
    #     dataset = UnalignedPoseNetDataset()
    #
    # else:
    #     raise ValueError("Dataset mode [%s] not recognized." % opt.dataset_mode)

    print("Dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)

        def init_fn(worker_id):
            torch.manual_seed(opt.seed)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads),
            worker_init_fn=init_fn,
#            collate_fn=custom_collate
        )

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
