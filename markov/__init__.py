"""Helpers for the Markov chain and HMM tutorials in this repository."""

from markov.absorbing import AbsorbingChain

__all__ = ["AbsorbingChain", "MarkovChain"]


def __getattr__(name):
    """Import MarkovChain only when it is asked for (PEP 562).

    MarkovChain draws pictures and so needs matplotlib, while absorbing.py
    is pure numpy. Importing the drawing code here at module level would
    make `from markov.absorbing import AbsorbingChain` fail on a checkout
    that has numpy but no plotting stack, which is enough to stop pytest
    from collecting the tests at all.
    """
    if name == "MarkovChain":
        from markov.markovchain_plot import MarkovChain

        return MarkovChain
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
