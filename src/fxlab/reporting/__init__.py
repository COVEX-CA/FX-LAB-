"""Informes HTML autocontenidos de un experimento de barrido.

Un informe de investigación existe para transmitir cuánto sabes, no para
lucir bien: cada elemento visual de este paquete está pensado para hacer
más difícil autoengañarse, no más fácil. De ahí el orden fijo de las
secciones (veredicto primero, curvas de equity al final), la escala de
color anclada en cero y discretizada en bandas, y la prohibición de
mostrar una tabla de "mejores configuraciones" sin el veredicto y el
recuento efectivo de pruebas en el mismo elemento.

Este paquete es **solo presentación**: no ejecuta backtests, no ejecuta
barridos y no recalcula ningún criterio de veredicto. Recibe los
resultados de `fxlab.validation` ya calculados y los dibuja.
"""

from fxlab.reporting.report import ReportInputs, build_report

__all__ = ["ReportInputs", "build_report"]
