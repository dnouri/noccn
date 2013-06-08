from .ccn import shownet
from .script import run_model


def console():
    run_model(shownet.ShowConvNet, 'show')
