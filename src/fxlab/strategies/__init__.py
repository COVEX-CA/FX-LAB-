"""Hipótesis de trading expresadas como generadores de señales.

Una estrategia en este paquete recibe precios y devuelve *señales*
(instantes de entrada y salida, con dirección) — nunca órdenes, tamaños de
posición, ni una decisión sobre si el resultado es bueno. Esa distinción
importa: la conversión de señales a operaciones (con costes) la hace el
motor de barrido (`fxlab.sweep`), y si el resultado es una ventaja real o
ruido lo decide la fase de validación estadística, que todavía no existe.
"""
