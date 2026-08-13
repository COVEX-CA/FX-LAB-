# AGENTS.md

Instrucciones para agentes de código (Codex, Claude Code) que trabajen en este repositorio.

---

## 1. Qué es este proyecto

`fx-research-lab` es un entorno de **investigación cuantitativa** sobre Forex.
No es un bot de trading. No ejecuta órdenes reales. No se conecta a ningún bróker.

Su objetivo es responder preguntas del tipo *"¿tiene esta hipótesis una ventaja
estadística real, o es ruido?"* con el rigor suficiente como para que la respuesta
sea creíble.

La primera pregunta bajo estudio es: **¿tiene edge el cruce de medias móviles en
Forex, y el tipo de media importa o es indiferente?**

El proyecto está diseñado para que esa pregunta sea intercambiable. No asumas que
el repositorio existe para cruces de medias.

---

## 2. Reglas de oro (innegociables)

Cualquier PR que viole una de estas reglas se rechaza, por muy bien escrito que esté.

1. **Nunca mires al futuro.** Ninguna señal en la barra `t` puede usar información
   de `t` o posterior. Las señales se calculan sobre barras cerradas y se ejecutan
   en la apertura de `t+1`. Si dudas, desplaza.
2. **Los costes siempre están puestos.** Ningún backtest se ejecuta sin spread y
   comisión. No existe el modo "sin costes para ir rápido".
3. **Toda prueba queda registrada.** Cada combinación de parámetros evaluada se
   escribe en el registro de trials, gane o pierda. El número total de pruebas es
   un dato imprescindible para corregir el sesgo de selección.
4. **El out-of-sample no se toca.** El último 25% del histórico está reservado.
   No se usa para desarrollar, ni para ajustar, ni para "echar un vistazo".
5. **Nada de datos en git.** Los parquet y los ficheros descargados no se versionan.
6. **No se seleccionan picos.** Los criterios de selección buscan mesetas de
   parámetros estables, no máximos aislados.

---

## 3. Stack

| Pieza | Elección |
|---|---|
| Python | 3.11 (fijo — numba/vectorbt van por detrás de las versiones nuevas) |
| Gestor | `uv` |
| Datos | `dukascopy-python` |
| Almacenamiento | Parquet vía `pyarrow` |
| Motor de backtest | `vectorbt` (open source) |
| Numérico | `numpy`, `pandas`, `scipy` |
| Registro de trials | SQLite (stdlib `sqlite3`) |
| Tests | `pytest` |
| Lint / formato | `ruff` |
| Tipos | `mypy` en modo estricto sobre `src/` |

No añadas dependencias nuevas sin justificarlo en la descripción del PR.

---

## 4. Estructura

```
src/fxlab/
├── data/         # descarga, caché en parquet, resampleo
├── indicators/   # wrappers de medias para IndicatorFactory
├── sweep/        # barrido de parámetros + registro de trials
├── validation/   # walk-forward, Deflated Sharpe Ratio, PBO
└── reporting/    # métricas, mapas de calor, informes
tests/            # espeja la estructura de src/
configs/          # YAML de experimentos (sin secretos)
scripts/          # entradas de línea de comandos
data/             # ignorado por git
```

Cada módulo tiene una responsabilidad. Si una función necesita saber de dos capas
a la vez, el diseño está mal — pregunta antes de escribirla.

---

## 5. Comandos

```bash
uv sync                  # instalar
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # formato
uv run mypy src/         # tipos
```

Los cuatro últimos deben pasar en verde antes de abrir un PR.

---

## 6. Convenciones de código

- Type hints en todas las firmas públicas. `mypy` estricto sobre `src/`.
- Docstrings estilo Google en funciones públicas. En español o inglés, pero
  consistente dentro de cada módulo.
- Nombres de variables en inglés. Comentarios y docstrings en español.
- Timestamps **siempre** en UTC y timezone-aware. Nunca naive.
- Precios en `float64`. Volúmenes en `float64`. Nada de `float32`.
- Sin `print()` en `src/`. Usa `logging`.
- Sin rutas absolutas ni rutas hardcodeadas. Todo vía `pathlib` y configuración.
- Funciones puras siempre que se pueda: entrada → salida, sin estado oculto.

---

## 7. Tests

- Cada módulo nuevo llega con sus tests. Un PR sin tests no está terminado.
- Los tests **no** descargan datos de internet. Usa fixtures pequeñas y fijas.
- Para lógica numérica, compara contra valores calculados a mano en el test, no
  contra la salida de la propia función.
- Test obligatorio en cualquier cosa que genere señales: un caso que falle si se
  introduce lookahead.

---

## 8. Cómo se trabaja aquí

- **Una tarea = un módulo = un PR.** No agrupes fases.
- Si una tarea parece necesitar tocar tres carpetas, está mal planteada: dilo en
  vez de intentarlo.
- Antes de escribir código, resume en dos frases qué vas a hacer y qué asumes.
- Si una especificación es ambigua, **pregunta**. No elijas una interpretación y
  sigas adelante. En este dominio una suposición silenciosa se convierte en un
  resultado falso que parece bueno.
- No refactorices código fuera del alcance de tu tarea.

---

## 9. Zonas prohibidas

No modifiques sin instrucción explícita:

- `AGENTS.md` (este fichero)
- `src/fxlab/validation/` una vez esté validado — es la capa que impide
  autoengañarse; cambiarla para que "salgan mejores números" invalida el proyecto
- El fichero de partición train/test
- Cualquier cosa bajo `data/`

---

## 10. Qué significa "terminado"

Un PR está listo cuando:

- [ ] `ruff check`, `ruff format --check`, `mypy` y `pytest` pasan
- [ ] Hay tests nuevos que cubren el código nuevo
- [ ] Ninguna regla de oro de la sección 2 se ha violado
- [ ] La descripción del PR explica **qué asumiste** y **qué no cubriste**
