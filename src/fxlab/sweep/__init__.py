"""Motor de barrido de parámetros y registro de pruebas.

Este paquete ejecuta combinaciones de parámetros de una estrategia sobre la
partición de desarrollo, siempre con costes de transacción activos, y
registra cada resultado — gane o pierda — en `fxlab.sweep.registry`. No
decide cuál es la "mejor" combinación: eso es la fase de validación
estadística, que todavía no existe.
"""
