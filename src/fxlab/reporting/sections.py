"""Constructores de cada sección del informe, en fragmentos de HTML.

Cada función devuelve una cadena de HTML ya lista para concatenar. Las
figuras van embebidas como `<div>` de Plotly sin la librería: `report.py`
inyecta `plotly.js` una sola vez, en línea, para que el fichero final no
dependa de la red al abrirlo.

## Decisiones de presentación, y por qué

- **Escala de color divergente anclada en cero y discretizada en bandas**
  (`_banded_diverging_colorscale`). Anclada en cero porque el Sharpe tiene
  signo y el cero significa algo: una rejilla enteramente negativa no puede
  mostrar zonas "brillantes" que parezcan buenas. Discretizada porque un
  gradiente continuo sobre ruido se lee como estructura — con bandas, una
  meseta es una mancha contigua de un color y un pico aislado es una celda
  suelta, que es exactamente la distinción que hay que poder hacer.
- **Límites simétricos** (`±max|valor|`). Nunca recortados por percentiles:
  recortar la escala magnifica visualmente diferencias pequeñas, que es lo
  que esta fase prohíbe explícitamente.
- **Sin marcas de tiempo de generación.** El informe no imprime en ninguna
  parte la fecha en que se generó, ni el `created_at` del registro. Es
  consecuencia directa de la regla de que ningún dato del holdout aparezca
  en la salida: cualquier fecha de reloj real es posterior al corte, y
  distinguir "fecha de dato" de "fecha de metadato" dentro del HTML sería
  frágil. La trazabilidad se cubre con `code_version` (hash de commit), que
  no lleva fecha.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
import plotly.io as pio
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from fxlab.split import DEVELOPMENT_END, DEVELOPMENT_START
from fxlab.validation.pbo import PBOResult
from fxlab.validation.report import Verdict, VerdictReport
from fxlab.validation.walk_forward import WalkForwardResult

MAX_CONFIGS_IN_CORRELATION_VIEW = 200
"""Tope de configuraciones dibujadas en la matriz de correlación y el
dendrograma. No es un criterio de calidad: una matriz de 20.000×20.000 no
cabe en memoria ni se puede leer en pantalla. Cuando la rejilla lo supera se
dibuja una muestra determinista y **se dice en el propio gráfico**. El
recuento efectivo de pruebas que se muestra sigue siendo el del veredicto,
calculado sobre la rejilla completa."""

N_COLOR_BANDS = 11
"""Bandas de la escala divergente. Impar para que haya una banda centrada
exactamente en cero."""

SAMPLE_EQUITY_CURVES = 20
"""Curvas de contexto (muestra aleatoria con semilla fija) en la sección de
equity, además de la destacada y sus vecinas."""

_VERDICT_COLORS = {
    Verdict.RUIDO: ("#7f1d1d", "#fee2e2"),
    Verdict.NO_CONCLUYENTE: ("#78350f", "#fef3c7"),
    Verdict.CANDIDATO: ("#14532d", "#dcfce7"),
}


def _figure_html(fig: go.Figure) -> str:
    """Fragmento HTML de una figura, sin incluir plotly.js."""
    return str(pio.to_html(fig, include_plotlyjs=False, full_html=False))


def _banded_diverging_colorscale(n_bands: int = N_COLOR_BANDS) -> list[list[object]]:
    """Escala RdBu discretizada en `n_bands` escalones.

    Se emite cada color con dos paradas (inicio y fin de su banda), que es
    la forma en que Plotly representa una escala discreta.
    """
    positions = [(i + 0.5) / n_bands for i in range(n_bands)]
    colors = pcolors.sample_colorscale("RdBu", positions)
    scale: list[list[object]] = []
    for i, color in enumerate(colors):
        scale.append([i / n_bands, color])
        scale.append([(i + 1) / n_bands, color])
    return scale


def _symmetric_limit(values: np.ndarray) -> float:
    """`max|valor|` finito, para límites de color simétricos. 1.0 si no hay
    ningún valor finito (evita una escala degenerada)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    limit = float(np.max(np.abs(finite)))
    return limit if limit > 0 else 1.0


def _esc(value: object) -> str:
    return html.escape(str(value))


# --- 2.0 Cobertura de datos ------------------------------------------------

_COVERAGE_TOLERANCE = pd.Timedelta(days=31)
"""Margen para decidir si un barrido cubre desarrollo entero. La primera y la
última barra reales caen a días del borde de la partición (festivos, fines de
semana, inicio del histórico disponible), no exactamente en él. Un mes absorbe
ese margen de disponibilidad sin dejar pasar un subperíodo deliberado, que
siempre se aparta meses o años del borde."""


def data_coverage_banner(trials: pd.DataFrame) -> str:
    """Aviso, arriba del todo, de qué tramo del histórico barrió el experimento.

    Si el barrido no cubre la partición de desarrollo entera, lo dice de
    forma prominente (no en una nota al pie): elegir un subperíodo es una
    forma de selección, y si pasa desapercibida es sesgo invisible. Cuando sí
    la cubre, deja una línea neutra de contexto. Nunca imprime el corte de
    holdout: el fin de desarrollo se muestra como su última fecha inclusive.
    """
    if not {"start_date", "end_date"}.issubset(trials.columns):
        return ""

    swept_start = pd.to_datetime(trials["start_date"], utc=True).min()
    swept_end = pd.to_datetime(trials["end_date"], utc=True).max()

    full_span = f"{DEVELOPMENT_START.date()} … {DEVELOPMENT_END.date()}"
    swept = f"{swept_start.date()} … {swept_end.date()}"

    covers_start = swept_start <= DEVELOPMENT_START + _COVERAGE_TOLERANCE
    covers_end = swept_end >= DEVELOPMENT_END - _COVERAGE_TOLERANCE
    if covers_start and covers_end:
        return (
            f'<p class="note" id="data-coverage">Cobertura de datos: el barrido abarca '
            f"la partición de desarrollo completa ({full_span}).</p>"
        )

    return (
        '<div class="collapsed-warning" id="data-coverage"><strong>Este barrido no '
        "cubre la partición de desarrollo completa.</strong> Se ha barrido solo "
        f"<strong>{swept}</strong>, un subconjunto de desarrollo ({full_span}). "
        "Elegir un subperíodo es en sí mismo una forma de selección: lo que se "
        "concluye abajo vale para ese tramo, no para todo el histórico de "
        "desarrollo, y no es directamente comparable con un barrido completo.</div>"
    )


# --- 2.1 Veredicto ---------------------------------------------------------


def verdict_section(verdict: VerdictReport, min_effective_trials: int) -> str:
    """Bloque de veredicto. Va el primero del informe, antes de todo gráfico."""
    text_color, background = _VERDICT_COLORS[verdict.verdict]
    headline = _esc(verdict.verdict.value.upper())

    collapsed_banner = ""
    if verdict.grid_collapsed:
        collapsed_banner = (
            '<p class="collapsed-warning"><strong>Veredicto limitado por colapso '
            "de rejilla.</strong> Solo "
            f"{verdict.n_trials_effective} pruebas efectivamente independientes "
            f"(mínimo exigido {min_effective_trials}). La rejilla no contiene "
            "suficientes apuestas independientes como para que el DSR aporte "
            "deflación, así que CANDIDATO no estaba disponible sea cual sea el "
            "número que muestre el DSR.</p>"
        )

    reasons = "".join(f"<li>{_esc(reason)}</li>" for reason in verdict.reasons)

    return f"""
<section id="verdict">
  <h2>1. Veredicto</h2>
  <div class="verdict-box" style="color:{text_color};background:{background};">
    <span class="verdict-label">{headline}</span>
  </div>
  {collapsed_banner}
  <table class="metrics">
    <tr><th>DSR</th><td>{verdict.dsr:.4f}</td></tr>
    <tr><th>PBO</th><td>{verdict.pbo:.4f}</td></tr>
    <tr><th>Pruebas (recuento bruto)</th><td>{verdict.n_trials_raw}</td></tr>
    <tr><th>Pruebas (recuento efectivo)</th><td>{verdict.n_trials_effective}</td></tr>
    <tr><th>Mínimo efectivo exigido</th><td>{min_effective_trials}</td></tr>
  </table>
  <h3>Motivos</h3>
  <ul class="reasons">{reasons}</ul>
</section>
"""


# --- 2.2 Diagnóstico de la rejilla -----------------------------------------


def _correlation_and_order(
    returns: pd.DataFrame, distance_threshold: float
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Matriz de correlación, matriz de enlace y orden de hojas.

    Usa exactamente el mismo método que
    `fxlab.validation.deflated_sharpe.effective_n_trials` (distancia
    `1 - |r|`, enlace promedio) para que el dendrograma que se dibuja sea el
    mismo del que sale el recuento efectivo del veredicto.
    """
    corr = np.nan_to_num(returns.corr().to_numpy(), nan=0.0)
    distance = 1 - np.abs(corr)
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2
    linkage_matrix = hierarchy.linkage(squareform(distance, checks=False), method="average")
    leaf_order = hierarchy.leaves_list(linkage_matrix)
    labels = [str(returns.columns[i]) for i in leaf_order]
    return corr, linkage_matrix, labels


def grid_diagnostics_section(
    all_returns: pd.DataFrame,
    distance_threshold: float,
    verdict: VerdictReport,
    min_effective_trials: int,
    seed: int = 0,
) -> str:
    """Calidad del experimento, antes de mostrar ningún resultado.

    Treinta variantes de la misma idea no son treinta pruebas: quien lea el
    informe tiene que ver eso antes que ninguna cifra de rentabilidad.
    """
    n_raw = all_returns.shape[1]

    sampled = all_returns
    sampling_note = ""
    if n_raw > MAX_CONFIGS_IN_CORRELATION_VIEW:
        rng = np.random.default_rng(seed)
        picked = np.sort(rng.choice(n_raw, MAX_CONFIGS_IN_CORRELATION_VIEW, replace=False))
        sampled = all_returns.iloc[:, picked]
        sampling_note = (
            f"<p class='note'>La rejilla tiene {n_raw} configuraciones: se dibuja "
            f"una muestra determinista de {MAX_CONFIGS_IN_CORRELATION_VIEW} "
            "porque la matriz completa no es legible ni cabe en memoria. El "
            "recuento efectivo del veredicto sí está calculado sobre la rejilla "
            "completa.</p>"
        )

    figures = ""
    mean_abs_corr = float("nan")
    if sampled.shape[1] >= 2:
        corr, linkage_matrix, labels = _correlation_and_order(sampled, distance_threshold)
        off_diagonal = corr[~np.eye(corr.shape[0], dtype=bool)]
        mean_abs_corr = float(np.abs(off_diagonal).mean())

        leaf_order = hierarchy.leaves_list(linkage_matrix)
        ordered = np.abs(corr)[np.ix_(leaf_order, leaf_order)]
        heat = go.Figure(
            go.Heatmap(
                z=ordered,
                x=labels,
                y=labels,
                colorscale="Greys",
                zmin=0.0,
                zmax=1.0,
                colorbar=dict(title="|r|"),
            )
        )
        heat.update_layout(
            title="Correlación absoluta entre configuraciones, ordenada por cluster",
            height=560,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
        )

        dendro = go.Figure()
        dendrogram_data = hierarchy.dendrogram(linkage_matrix, no_plot=True)
        for xs, ys in zip(dendrogram_data["icoord"], dendrogram_data["dcoord"], strict=True):
            dendro.add_trace(
                go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#334155", width=1))
            )
        dendro.add_hline(
            y=distance_threshold,
            line=dict(color="#b91c1c", dash="dash"),
            annotation_text=f"corte {distance_threshold}",
        )
        dendro.update_layout(
            title="Dendrograma (distancia 1 - |r|, enlace promedio)",
            showlegend=False,
            height=380,
            xaxis=dict(showticklabels=False, title="configuraciones"),
            yaxis=dict(title="distancia", rangemode="tozero"),
        )
        figures = _figure_html(heat) + _figure_html(dendro)

    warning = ""
    if verdict.grid_collapsed:
        warning = (
            "<p class='collapsed-warning'><strong>Aviso: rejilla colapsada.</strong> "
            f"{verdict.n_trials_effective} pruebas efectivas frente a un mínimo de "
            f"{min_effective_trials}. Las configuraciones de esta rejilla son en su "
            "mayoría variantes de una misma apuesta, no pruebas distintas.</p>"
        )

    corr_text = "n/d" if np.isnan(mean_abs_corr) else f"{mean_abs_corr:.3f}"
    return f"""
<section id="grid-diagnostics">
  <h2>2. Diagnóstico de la rejilla</h2>
  <p>Antes que ningún resultado: cuántas apuestas realmente distintas contiene
  este experimento. Configuraciones que caen en el mismo cluster cuentan como
  una sola prueba a efectos de corregir el sesgo de selección.</p>
  {warning}
  <table class="metrics">
    <tr><th>Configuraciones (bruto)</th><td>{verdict.n_trials_raw}</td></tr>
    <tr><th>Pruebas efectivas</th><td>{verdict.n_trials_effective}</td></tr>
    <tr><th>Mínimo exigido</th><td>{min_effective_trials}</td></tr>
    <tr><th>Correlación media |r| (muestra dibujada)</th><td>{corr_text}</td></tr>
  </table>
  {sampling_note}
  {figures}
</section>
"""


# --- 2.3 Superficie de parámetros ------------------------------------------


def _varying_parameters(params: pd.DataFrame) -> list[str]:
    return [column for column in params.columns if params[column].nunique(dropna=False) > 1]


def parameter_surface_section(
    trials: pd.DataFrame, params: pd.DataFrame, metric: str, best_index: int
) -> str:
    """Mapas de calor de `metric` frente a pares de parámetros.

    Los parámetros que no están en los ejes se fijan en los valores de la
    configuración mejor puntuada, y ese corte se etiqueta en el título de
    cada gráfico. Nunca se promedia sobre una dimensión: promediar esconde
    justo lo que hay que ver, que es si un máximo está acompañado o solo.
    """
    varying = _varying_parameters(params)
    reading_note = """
  <p class="reading">Cómo se lee este gráfico: una <strong>meseta</strong> —una
  zona amplia en la que los parámetros vecinos también funcionan— es señal de
  robustez, porque significa que el resultado no depende de haber acertado un
  valor exacto. Un <strong>pico aislado</strong>, una celda buena rodeada de
  celdas malas, es lo que produce el ruido cuando se prueban muchas
  combinaciones: no sobrevive a un cambio pequeño de parámetros ni a un tramo
  distinto del histórico. La escala está anclada en cero y discretizada en
  bandas precisamente para que la diferencia entre las dos situaciones se vea
  a simple vista y no dependa de un gradiente suave.</p>
"""

    if len(varying) < 2:
        return f"""
<section id="parameter-surface">
  <h2>3. Superficie de parámetros</h2>
  {reading_note}
  <p class="note">La rejilla varía menos de dos parámetros
  ({len(varying)}), así que no hay ninguna superficie bidimensional que
  dibujar. No es un error: es una rejilla que no explora un plano.</p>
</section>
"""

    values = pd.to_numeric(trials[metric], errors="coerce").to_numpy(dtype="float64")
    limit = _symmetric_limit(values)
    colorscale = _banded_diverging_colorscale()

    figures = []
    for i in range(len(varying)):
        for j in range(i + 1, len(varying)):
            x_param, y_param = varying[i], varying[j]
            fixed = [p for p in varying if p not in (x_param, y_param)]

            mask = pd.Series(True, index=params.index)
            for parameter in fixed:
                mask &= params[parameter] == params[parameter].iloc[best_index]
            slice_params = params[mask]
            slice_values = pd.Series(values, index=params.index)[mask]

            pivot = pd.DataFrame(
                {
                    "x": slice_params[x_param].astype(str),
                    "y": slice_params[y_param].astype(str),
                    "v": slice_values.to_numpy(),
                }
            ).pivot_table(index="y", columns="x", values="v", aggfunc="first")

            fixed_label = (
                ", ".join(f"{p}={params[p].iloc[best_index]}" for p in fixed)
                if fixed
                else "sin más parámetros que fijar"
            )
            fig = go.Figure(
                go.Heatmap(
                    z=pivot.to_numpy(),
                    x=[str(c) for c in pivot.columns],
                    y=[str(r) for r in pivot.index],
                    colorscale=colorscale,
                    zmin=-limit,
                    zmid=0.0,
                    zmax=limit,
                    colorbar=dict(title=metric),
                )
            )
            fig.update_layout(
                title=(
                    f"{metric}: {y_param} frente a {x_param}<br>"
                    f"<sub>corte con {fixed_label} (valores de la mejor puntuada; "
                    f"sin promediar ninguna dimensión)</sub>"
                ),
                height=460,
                xaxis=dict(title=x_param, type="category"),
                yaxis=dict(title=y_param, type="category"),
            )
            figures.append(_figure_html(fig))

    return f"""
<section id="parameter-surface">
  <h2>3. Superficie de parámetros</h2>
  {reading_note}
  {"".join(figures)}
</section>
"""


# --- 2.4 Walk-forward ------------------------------------------------------


def walk_forward_section(walk_forward: WalkForwardResult, verdict: VerdictReport) -> str:
    """Entrenamiento frente a prueba por pliegue, y estabilidad de los
    parámetros elegidos a lo largo del tiempo."""
    folds = walk_forward.folds
    if not folds:
        return """
<section id="walk-forward">
  <h2>4. Walk-forward</h2>
  <p class="note">El walk-forward no produjo ningún pliegue evaluable.</p>
</section>
"""

    fold_labels = [f"{i + 1}" for i in range(len(folds))]
    train_values = [
        f.train_result.sharpe_annualized if f.train_result is not None else None for f in folds
    ]
    test_values = [
        f.test_result.sharpe_annualized if f.test_result is not None else None for f in folds
    ]

    metrics_fig = go.Figure()
    metrics_fig.add_trace(
        go.Bar(x=fold_labels, y=train_values, name="entrenamiento", marker_color="#94a3b8")
    )
    metrics_fig.add_trace(
        go.Bar(x=fold_labels, y=test_values, name="prueba (out-of-sample)", marker_color="#1d4ed8")
    )
    metrics_fig.add_hline(y=0, line=dict(color="#111827", width=1))
    metrics_fig.update_layout(
        title="Sharpe anualizado por pliegue: entrenamiento frente a prueba",
        barmode="group",
        height=420,
        xaxis=dict(title="pliegue"),
        yaxis=dict(title="Sharpe anualizado"),
    )

    parameter_rows = ""
    stability_html = ""
    chosen = [f.best_params.as_dict() if f.best_params is not None else {} for f in folds]
    parameter_names = sorted({key for entry in chosen for key in entry})
    if parameter_names:
        stability_fig = go.Figure()
        for name in parameter_names:
            stability_fig.add_trace(
                go.Scatter(
                    x=fold_labels,
                    y=[str(entry.get(name, "—")) for entry in chosen],
                    mode="lines+markers",
                    name=name,
                )
            )
        stability_fig.update_layout(
            title="Parámetros elegidos en cada pliegue",
            height=420,
            xaxis=dict(title="pliegue"),
            yaxis=dict(title="valor elegido", type="category"),
        )
        stability_html = _figure_html(stability_fig)

        for name in parameter_names:
            cells = "".join(f"<td>{_esc(entry.get(name, '—'))}</td>" for entry in chosen)
            parameter_rows += f"<tr><th>{_esc(name)}</th>{cells}</tr>"

    degradation = verdict.walk_forward_mean_degradation_annualized
    degradation_text = "n/d" if degradation is None else f"{degradation:.4f}"

    header_cells = "".join(f"<th>{label}</th>" for label in fold_labels)
    return f"""
<section id="walk-forward">
  <h2>4. Walk-forward</h2>
  {_figure_html(metrics_fig)}
  <h3>Estabilidad de los parámetros elegidos</h3>
  <p>Si los parámetros ganadores cambian radicalmente de un pliegue al
  siguiente, no hay estabilidad: la optimización está siguiendo el ruido de
  cada ventana en vez de una propiedad persistente del mercado.</p>
  {stability_html}
  <table class="metrics folds">
    <tr><th>pliegue</th>{header_cells}</tr>
    {parameter_rows}
  </table>
  <h3>Degradación agregada</h3>
  <p class="note"><strong>Informativa, no vinculante.</strong> La degradación
  media (entrenamiento menos prueba) es <strong>{degradation_text}</strong>.
  No entra en ningún criterio del veredicto porque no es adimensional: una
  degradación de 0.05 sobre un Sharpe de entrenamiento de 0.1 es catastrófica
  y sobre uno de 2.0 es irrelevante, así que un umbral fijo no significaría lo
  mismo en distintas zonas de la rejilla. La única señal walk-forward que sí
  puentea el veredicto es el signo del Sharpe medio de prueba.</p>
</section>
"""


# --- 2.5 Distribución de λ del PBO -----------------------------------------


def pbo_lambda_section(pbo: PBOResult | None) -> str:
    """Histograma de los logits, con la fracción a la izquierda de cero
    señalada: esa fracción **es** el PBO, no un resumen de él.

    `pbo=None` cuando la rejilla tiene una sola configuración: el CSCV
    necesita al menos dos para poder rankear, y con una sola no hay ninguna
    selección que pueda estar sobreajustada. Se dice explícitamente en vez
    de omitir la sección.
    """
    if pbo is None:
        return """
<section id="pbo-lambda">
  <h2>5. Distribución de λ del PBO</h2>
  <p class="note">El experimento tiene una sola configuración, así que el PBO
  no está definido: el CSCV mide con qué frecuencia la ganadora dentro de
  muestra cae en la mitad inferior fuera de muestra, y con una única
  configuración no hay ni ranking ni selección que pueda estar sobreajustada.
  Su ausencia no es un fallo del experimento, pero sí significa que esta vía
  de evidencia no aporta nada aquí.</p>
</section>
"""

    logits = np.asarray(pbo.logits, dtype="float64")
    finite = logits[np.isfinite(logits)]
    below = finite[finite < 0]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=below,
            name="λ < 0 (la ganadora cayó en la mitad inferior)",
            marker_color="#b91c1c",
            nbinsx=60,
        )
    )
    fig.add_trace(
        go.Histogram(x=finite[finite >= 0], name="λ ≥ 0", marker_color="#94a3b8", nbinsx=60)
    )
    fig.add_vline(x=0, line=dict(color="#111827", width=2))
    fig.update_layout(
        title=(
            f"Distribución de λ sobre {pbo.n_combinations} combinaciones "
            f"(S={pbo.s}, N={pbo.n_configurations})<br>"
            f"<sub>la fracción a la izquierda de la línea es el PBO = {pbo.pbo:.4f}</sub>"
        ),
        barmode="overlay",
        height=440,
        xaxis=dict(title="λ = ln(ω / (1-ω))"),
        yaxis=dict(title="combinaciones", rangemode="tozero"),
    )

    return f"""
<section id="pbo-lambda">
  <h2>5. Distribución de λ del PBO</h2>
  <p>Cada combinación in-sample/out-of-sample aporta un λ. Los que quedan a la
  izquierda de cero son aquellos en los que la configuración ganadora dentro de
  muestra acabó en la mitad inferior fuera de muestra. Su fracción sobre el
  total es literalmente el PBO: <strong>{len(below)} de {len(finite)} =
  {pbo.pbo:.4f}</strong>. Un PBO cercano a 0.5 significa que el proceso de
  selección no distingue nada del azar.</p>
  {_figure_html(fig)}
</section>
"""


# --- 2.6 Curvas de equity --------------------------------------------------


@dataclass(frozen=True)
class RankedConfigurations:
    """Configuraciones mejor puntuadas, inseparables de su contexto.

    Nunca se expone una tabla de "mejores" a secas: el veredicto y el
    recuento efectivo de pruebas viajan en el mismo objeto, porque un
    ranking sin ellos invita exactamente al error que este proyecto intenta
    evitar.
    """

    table: pd.DataFrame
    verdict: Verdict
    n_trials_raw: int
    n_trials_effective: int
    grid_collapsed: bool


def rank_configurations(
    trials: pd.DataFrame, metric: str, verdict: VerdictReport, top_n: int
) -> RankedConfigurations:
    """Las `top_n` configuraciones mejor puntuadas, con el veredicto adjunto."""
    ordered = trials.assign(_metric=pd.to_numeric(trials[metric], errors="coerce")).sort_values(
        "_metric", ascending=False, na_position="last"
    )
    return RankedConfigurations(
        table=ordered.head(top_n).drop(columns="_metric"),
        verdict=verdict.verdict,
        n_trials_raw=verdict.n_trials_raw,
        n_trials_effective=verdict.n_trials_effective,
        grid_collapsed=verdict.grid_collapsed,
    )


def _neighbour_indices(params: pd.DataFrame, best_index: int) -> list[int]:
    """Configuraciones que difieren de la destacada en un solo parámetro, y
    en un solo paso dentro de los valores ordenados de ese parámetro."""
    neighbours: list[int] = []
    best = params.iloc[best_index]
    for column in params.columns:
        uniques = sorted(params[column].dropna().unique().tolist(), key=str)
        if len(uniques) < 2 or best[column] not in uniques:
            continue
        position = uniques.index(best[column])
        wanted = [uniques[p] for p in (position - 1, position + 1) if 0 <= p < len(uniques)]
        others = [c for c in params.columns if c != column]
        for value in wanted:
            mask = params[column] == value
            for other in others:
                mask &= params[other] == best[other]
            neighbours.extend(params.index[mask].tolist())
    return sorted(set(neighbours) - {best_index})


def equity_curves_section(
    all_returns: pd.DataFrame,
    params: pd.DataFrame,
    ranked: RankedConfigurations,
    best_index: int,
    seed: int = 0,
) -> str:
    """Curvas de equity de un subconjunto representativo: la destacada, sus
    vecinas en el espacio de parámetros, y una muestra de contexto.

    El punto de mostrar las vecinas es poder ver si la destacada está sola o
    acompañada. Una curva que se despega de sus vecinas inmediatas es un pico
    aislado, aunque su número sea el mejor de la tabla.
    """
    equity = (1 + all_returns).cumprod()
    n_configs = all_returns.shape[1]
    neighbours = _neighbour_indices(params, best_index)

    rng = np.random.default_rng(seed)
    remaining = [i for i in range(n_configs) if i != best_index and i not in neighbours]
    sample_size = min(SAMPLE_EQUITY_CURVES, len(remaining))
    sampled = (
        sorted(rng.choice(remaining, sample_size, replace=False).tolist()) if sample_size else []
    )

    fig = go.Figure()
    for index in sampled:
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity.iloc[:, index],
                mode="lines",
                line=dict(color="#cbd5e1", width=1),
                name="resto (muestra)",
                showlegend=index == sampled[0],
                hoverinfo="skip",
            )
        )
    for index in neighbours:
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity.iloc[:, index],
                mode="lines",
                line=dict(color="#f59e0b", width=1.5),
                name="vecinas en parámetros",
                showlegend=index == neighbours[0],
            )
        )
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.iloc[:, best_index],
            mode="lines",
            line=dict(color="#1d4ed8", width=3),
            name="mejor puntuada",
        )
    )
    fig.update_layout(
        title="Curvas de equity (capital acumulado, base 1)",
        height=520,
        xaxis=dict(title="fecha"),
        yaxis=dict(title="capital acumulado"),
    )

    collapsed = (
        " La rejilla está colapsada, así que este ranking abarca menos apuestas "
        "independientes de las que sugiere su longitud."
        if ranked.grid_collapsed
        else ""
    )

    header = "".join(f"<th>{_esc(c)}</th>" for c in ranked.table.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{_esc(v)}</td>" for v in row) + "</tr>"
        for row in ranked.table.itertuples(index=False)
    )

    return f"""
<section id="equity">
  <h2>6. Curvas de equity</h2>
  <p>Se muestran juntas la configuración mejor puntuada, sus
  {len(neighbours)} vecinas inmediatas en el espacio de parámetros, y una
  muestra de {len(sampled)} configuraciones más como contexto. Si la
  destacada se despega de sus vecinas, es un pico aislado.</p>
  {_figure_html(fig)}
  <h3>Configuraciones mejor puntuadas</h3>
  <p class="ranking-context"><strong>Veredicto: {_esc(ranked.verdict.value.upper())}</strong>
  — {ranked.n_trials_raw} pruebas brutas, {ranked.n_trials_effective} efectivas.
  Este ranking no dice cuál es "la mejor estrategia": dice qué configuraciones
  puntuaron más alto dentro de un experimento cuya conclusión global es la de
  arriba.{collapsed}</p>
  <table class="metrics ranked"><tr>{header}</tr>{rows}</table>
</section>
"""
