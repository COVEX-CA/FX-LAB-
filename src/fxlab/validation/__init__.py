"""Validación estadística: decide si un resultado del barrido es una ventaja
real o ruido con la suficiente rigidez como para no engañarse.

Zona prohibida una vez validada (ver `AGENTS.md`, sección 9): no se modifica
para que "salgan mejores números". Nada en este paquete decide "la mejor
estrategia" — devuelve si el conjunto de resultados sostiene una conclusión,
categórica y con los umbrales fijados antes de ver ningún resultado real.
"""
