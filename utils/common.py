import sys
import os

class Logger(object):

    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def print_network(net):
    """
    print model architecture and numbers of parameters
    :param net:u'r model
    :return: None
    """
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)