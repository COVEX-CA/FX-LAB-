"""Modelo de costes de transacción: spread real + comisión, siempre activos.

Ni este módulo ni `fxlab.sweep.engine` tienen ningún parámetro ni ruta de
código que permita ejecutar un backtest sin costes:

- El **spread** se toma de la diferencia real entre bid y ask de los
  propios datos descargados, nunca de un valor fijo inventado. No es un
  parámetro configurable — se deriva de los datos en cada barra. Toda
  compra (abrir un largo, o cerrar un corto) ejecuta al precio **ask**;
  toda venta (abrir un corto, o cerrar un largo) ejecuta al precio
  **bid**. Es la única forma correcta de modelar spread a partir de datos
  bid/ask reales, no una elección entre varias.
- La **comisión** (`CostModel.commission`) es un coste adicional,
  proporcional al valor de cada operación. Es un campo obligatorio, sin
  valor por defecto en el código: cada configuración de barrido debe
  fijarlo explícitamente. Puede ser 0.0 si de verdad se quiere modelar un
  bróker sin comisión aparte (el spread real, que nunca es cero, se sigue
  aplicando siempre) — pero tiene que ser una decisión explícita de quien
  configura el experimento, nunca un valor por omisión silencioso.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """Modelo de costes de un experimento de barrido.

    Args:
        commission: proporción del valor de cada operación (p.ej. 0.00007
            para 0.7 puntos básicos). Sin valor por defecto a propósito.
    """

    commission: float

    def __post_init__(self) -> None:
        if self.commission < 0:
            raise ValueError(f"commission no puede ser negativa: {self.commission}")


def execution_prices(
    long_entries: pd.Series,
    long_exits: pd.Series,
    short_entries: pd.Series,
    short_exits: pd.Series,
    bid_close: pd.Series,
    ask_close: pd.Series,
) -> pd.Series:
    """Precio de ejecución barra a barra: ask al comprar, bid al vender.

    - Comprar = abrir un largo (`long_entries`) o cerrar un corto
      (`short_exits`): ejecuta al `ask_close` de esa barra.
    - Vender = abrir un corto (`short_entries`) o cerrar un largo
      (`long_exits`): ejecuta al `bid_close` de esa barra.

    Las barras sin ninguna señal no tienen orden que ejecutar, así que su
    valor en la serie devuelta no se usa (vectorbt solo consulta el precio
    donde hay señal); se rellenan con `bid_close` por continuidad, no
    porque tengan significado.

    Todas las series deben compartir el mismo índice.
    """
    buys = long_entries | short_exits
    sells = short_entries | long_exits

    price = bid_close.copy()
    price[buys] = ask_close[buys]
    price[sells] = bid_close[sells]
    return price
