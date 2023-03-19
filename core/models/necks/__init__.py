from .DoNothing import *
from .simple_neck import SimpleNeck

def neck_entry(config):
    return globals()[config['type']](**config['kwargs'])
