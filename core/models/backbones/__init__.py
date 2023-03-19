from .vit import vit_base_patch16

def backbone_entry(config):
    return globals()[config['type']](**config['kwargs'])
