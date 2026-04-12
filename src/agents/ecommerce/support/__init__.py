from __future__ import annotations


def construct_graph():
    from .agent import construct_graph as _construct_graph
    return _construct_graph()


def invoke_model(state):
    from .agent import invoke_model as _invoke_model
    return _invoke_model(state)


def main(**kwargs):
    from .agent import main as _main
    return _main(**kwargs)
