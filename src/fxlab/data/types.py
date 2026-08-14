"""Tipos compartidos entre los submódulos de `fxlab.data`."""

from __future__ import annotations

from enum import Enum


class OfferSide(Enum):
    """Lado de la cotización. Bid y ask se descargan y cachean por separado:
    el spread real (necesario para modelar costes) se calcula a partir de ambos.
    """

    BID = "bid"
    ASK = "ask"
