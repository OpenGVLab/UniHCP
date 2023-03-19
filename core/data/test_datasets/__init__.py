from .images.reid_dataset import ReIDTestDataset, ReIDTestDatasetDev

def dataset_entry(config):
    # print('config[kwargs]',config['kwargs'])
    return globals()[config['type']](**config['kwargs'])
