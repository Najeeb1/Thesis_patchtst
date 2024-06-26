from .dataloader_taxibj import load_data as load_taxibj
from .dataloader_moving_mnist import load_data as load_mmnist
# from .dataloader_kth import load_data as load_kth
# from .dataloader_kitticaltech import load_data as load_kittical

def load_data(dataname,batch_size, val_batch_size, data_root, num_workers, **kwargs):
    if dataname == 'mmnist':
        return load_mmnist(batch_size, val_batch_size, data_root, num_workers)
    # elif dataname == 'taxibj':
    #     return load_mmnist(batch_size, val_batch_size, data_root, num_workers)
    # elif dataname == 'mmnist':
    #     return load_mmnist(batch_size, val_batch_size, data_root, num_workers)
    # elif dataname == 'kth':
    #     return load_kth(batch_size, val_batch_size, data_root, num_workers)
    # elif dataname == 'kitti_caltech':
    #     return load_kittical(batch_size, val_batch_size, data_root, num_workers)
    