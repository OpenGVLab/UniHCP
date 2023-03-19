from .network import AIOHead

def decoder_entry(config):
    return globals()[config['type']](**config['kwargs'])
