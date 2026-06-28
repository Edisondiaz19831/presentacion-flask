import os
import sqlite3
import pandas as pd
from flask import Flask, request, render_template_string, redirect, url_for

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ahora la base de datos está en el directorio principal
DB_PATH = os.path.join(BASE_DIR, "markov.db")

DEDOS = [1, 2, 3, 4, 5]
EPSILON = 0.001

app = None  # Cambiado de Flask(__name__) a None


def conectar():
    return sqlite3.connect(DB_PATH)


def parse_melodia(texto):
    return [int(x.strip()) for x in texto.split(",") if x.strip()]


def es_tecla_negra(midi):
    return midi % 12 in [1, 3, 6, 8, 10]


def direccion(delta):
    if delta is None:
        return "none"
    if delta > 0:
        return "up"
    if delta < 0:
        return "down"
    return "repeat"


def size_salto(delta):
    if delta is None:
        return "none"
    d = abs(delta)
    if d == 0:
        return "repeat"
    if d <= 2:
        return "step"
    if d <= 5:
        return "skip"
    return "leap"


def movimiento(delta):
    if delta is None:
        return "none"
    size = size_salto(delta)
    dir_ = direccion(delta)
    if size == "repeat":
        return "repeat"
    return f"{size}_{dir_}"


def calcular_run(mov_prev, mov_next, dir_prev, dir_next, size_prev, size_next, is_first, is_last):
    if is_first or is_last:
        return "edge"
    if size_prev == "repeat" or size_next == "repeat":
        return "repeat_context"
    if (
        size_prev == "step"
        and size_next == "step"
        and dir_prev == dir_next
        and dir_prev in ["up", "down"]
    ):
        return "run_step_same_dir"
    if dir_prev != dir_next and dir_prev != "none" and dir_next != "none":
        return "change_dir"
    return "other"


def construir_ctx_key(mov_prev, mov_next, dir_prev, dir_next, size_prev, size_next, black, run):
    return (
        f"mp={mov_prev}"
        f"|mn={mov_next}"
        f"|dir_prev={dir_prev}"
        f"|dir_next={dir_next}"
        f"|size_prev={size_prev}"
        f"|size_next={size_next}"
        f"|black={black}"
        f"|run={run}"
    )


def construir_contextos(midis):
    eventos = []
    for i, midi in enumerate(midis):
        prev_midi = midis[i - 1] if i > 0 else None
        next_midi = midis[i + 1] if i < len(midis) - 1 else None
        delta_prev = midi - prev_midi if prev_midi is not None else None
        delta_next = next_midi - midi if next_midi is not None else None
        is_first = 1 if i == 0 else 0
        is_last = 1 if i == len(midis) - 1 else 0
        dir_prev = direccion(delta_prev)
        dir_next = direccion(delta_next)
        size_prev = size_salto(delta_prev)
        size_next = size_salto(delta_next)
        mov_prev = movimiento(delta_prev)
        mov_next = movimiento(delta_next)
        black = 1 if es_tecla_negra(midi) else 0
        run = calcular_run(mov_prev, mov_next, dir_prev, dir_next, size_prev, size_next, is_first, is_last)
        ctx_key = construir_ctx_key(mov_prev, mov_next, dir_prev, dir_next, size_prev, size_next, black, run)
        eventos.append({
            "t": i + 1,
            "midi": midi,
            "prev_midi": prev_midi,
            "next_midi": next_midi,
            "delta_prev": delta_prev,
            "delta_next": delta_next,
            "dir_prev": dir_prev,
            "dir_next": dir_next,
            "mov_prev": mov_prev,
            "mov_next": mov_next,
            "size_prev": size_prev,
            "size_next": size_next,
            "is_black_key": black,
            "is_first": is_first,
            "is_last": is_last,
            "run": run,
            "ctx_key": ctx_key,
        })
    return eventos


def cargar_base_emisiones():
    conn = conectar()
    df = pd.read_sql_query("""
    SELECT finger, ctx_key
    FROM dataset_event
    WHERE finger IS NOT NULL
      AND finger IN (1,2,3,4,5)
      AND ctx_key IS NOT NULL
    """, conn)
    conn.close()
    df["finger"] = df["finger"].astype(int)
    ctx_vocab = sorted(df["ctx_key"].dropna().unique())
    k = len(ctx_vocab)
    conteos = df.groupby(["finger", "ctx_key"]).size().reset_index(name="conteo")
    total_dedo = df.groupby("finger").size().reset_index(name="total_dedo")
    return df, conteos, total_dedo, ctx_vocab, k


def cargar_base_transiciones():
    conn = conectar()
    df = pd.read_sql_query("""
    SELECT prev_finger, finger
    FROM dataset_event
    WHERE prev_finger IS NOT NULL
      AND finger IS NOT NULL
      AND prev_finger IN (1,2,3,4,5)
      AND finger IN (1,2,3,4,5)
    """, conn)
    conn.close()
    df["prev_finger"] = df["prev_finger"].astype(int)
    df["finger"] = df["finger"].astype(int)
    conteos = df.groupby(["prev_finger", "finger"]).size().reset_index(name="conteo")
    total_origen = df.groupby("prev_finger").size().reset_index(name="total_origen")
    return df, conteos, total_origen


def prob_emision(dedo, ctx_key, conteos, total_dedo, k):
    total_row = total_dedo.loc[total_dedo["finger"] == dedo, "total_dedo"]
    if total_row.empty:
        return 0, 0, 0, 0, 0
    total = int(total_row.iloc[0])
    conteo_row = conteos[(conteos["finger"] == dedo) & (conteos["ctx_key"] == ctx_key)]
    conteo = int(conteo_row["conteo"].iloc[0]) if not conteo_row.empty else 0
    numerador = conteo + EPSILON
    denominador = total + EPSILON * k
    prob = numerador / denominador if denominador > 0 else 0
    return conteo, total, numerador, denominador, prob


def prob_transicion(dedo_anterior, dedo_actual, conteos, total_origen):
    total_row = total_origen.loc[total_origen["prev_finger"] == dedo_anterior, "total_origen"]
    if total_row.empty:
        return 0, 0, 0, 0, 0
    total = int(total_row.iloc[0])
    conteo_row = conteos[(conteos["prev_finger"] == dedo_anterior) & (conteos["finger"] == dedo_actual)]
    conteo = int(conteo_row["conteo"].iloc[0]) if not conteo_row.empty else 0
    numerador = conteo + EPSILON
    denominador = total + EPSILON * 5
    prob = numerador / denominador if denominador > 0 else 0
    return conteo, total, numerador, denominador, prob


def construir_prior_manual(dedo_inicial):
    dedo_inicial = int(dedo_inicial)
    return {d: 1.0 if d == dedo_inicial else 0.0 for d in DEDOS}


def buscar_secuencias_historicas(midis, dedo_inicial=None):
    conn = conectar()
    df = pd.read_sql_query("""
    SELECT id, dataset_id, dataset_name, "order", midi, finger
    FROM dataset_event
    WHERE midi IS NOT NULL
      AND finger IS NOT NULL
      AND finger IN (1,2,3,4,5)
    ORDER BY dataset_id, "order"
    """, conn)
    conn.close()

    coincidencias = []
    n = len(midis)

    for dataset_id, grupo in df.groupby("dataset_id"):
        grupo = grupo.sort_values("order").reset_index(drop=True)
        lista_midis = grupo["midi"].astype(int).tolist()
        lista_fingers = grupo["finger"].astype(int).tolist()

        for i in range(0, len(lista_midis) - n + 1):
            ventana = lista_midis[i:i+n]
            if ventana == midis:
                dedos_lista = lista_fingers[i:i+n]
                if dedo_inicial is not None and dedos_lista[0] != int(dedo_inicial):
                    continue
                sub = grupo.iloc[i:i+n]
                dedos = ",".join(map(str, dedos_lista))
                coincidencias.append({
                    "dataset_id": dataset_id,
                    "dataset_name": sub["dataset_name"].iloc[0],
                    "order_inicio": int(sub["order"].iloc[0]),
                    "order_fin": int(sub["order"].iloc[-1]),
                    "midis": ",".join(map(str, ventana)),
                    "dedos_lista": dedos_lista,
                    "dedos_historicos": dedos,
                })

    agrupadas = {}
    for c in coincidencias:
        clave = c["dedos_historicos"]
        if clave not in agrupadas:
            agrupadas[clave] = {
                "dedos_historicos": clave,
                "dedos_lista": c["dedos_lista"],
                "midis": c["midis"],
                "total": 0,
                "datasets": []
            }
        agrupadas[clave]["total"] += 1
        agrupadas[clave]["datasets"].append(c)

    return list(agrupadas.values())


def opciones_historicas_por_t(secuencias_historicas, idx_t):
    opciones = {}
    for grupo in secuencias_historicas:
        dedos_lista = grupo["dedos_lista"]
        if idx_t < len(dedos_lista):
            dedo = dedos_lista[idx_t]
            opciones.setdefault(dedo, 0)
            opciones[dedo] += grupo["total"]
    return opciones


def parse_override(texto):
    # formato: "4:1,5:2" => en t=4 usar dedo 1, en t=5 usar dedo 2
    overrides = {}
    if not texto:
        return overrides
    for parte in texto.split(","):
        parte = parte.strip()
        if not parte or ":" not in parte:
            continue
        t, d = parte.split(":", 1)
        try:
            overrides[int(t)] = int(d)
        except Exception:
            pass
    return overrides


def serialize_override(overrides):
    return ",".join(f"{t}:{d}" for t, d in sorted(overrides.items()))


def calcular_hmm_local(dedo_anterior, evento, conteos_tr, total_origen, conteos_em, total_dedo, k_ctx):
    candidatos = []
    for dedo_candidato in DEDOS:
        conteo_t, total_t, num_t, den_t, a = prob_transicion(dedo_anterior, dedo_candidato, conteos_tr, total_origen)
        conteo_e, total_e, num_e, den_e, b = prob_emision(dedo_candidato, evento["ctx_key"], conteos_em, total_dedo, k_ctx)
        score = a * b
        candidatos.append({
            "dedo_candidato": dedo_candidato,
            "a_transicion": a,
            "b_emision": b,
            "score": score,
            "conteo_transicion": conteo_t,
            "conteo_emision": conteo_e,
        })
    candidatos = sorted(candidatos, key=lambda r: r["score"], reverse=True)
    return candidatos


def ejecutar_asistido(midis, dedo_inicial, overrides):
    eventos = construir_contextos(midis)
    secuencias_historicas = buscar_secuencias_historicas(midis, dedo_inicial)

    df_em, conteos_em, total_dedo, ctx_vocab, k_ctx = cargar_base_emisiones()
    df_tr, conteos_tr, total_origen = cargar_base_transiciones()
    ctx_set = set(ctx_vocab)

    for e in eventos:
        e["existe_ctx"] = e["ctx_key"] in ctx_set

    ruta = [dedo_inicial]
    pasos = []
    tabla_detalle = []
    conflicto_pendiente = None

    dedo_anterior = dedo_inicial

    # t=1 ya está fijado por prior manual
    pasos.append({
        "t": 1,
        "midi": eventos[0]["midi"] if eventos else None,
        "hmm": dedo_inicial,
        "historicas": {dedo_inicial: 1},
        "elegido": dedo_inicial,
        "decision": "prior manual",
        "estado": "fijado"
    })

    for idx_evento in range(1, len(eventos)):
        e = eventos[idx_evento]
        t = e["t"]

        candidatos = calcular_hmm_local(
            dedo_anterior,
            e,
            conteos_tr,
            total_origen,
            conteos_em,
            total_dedo,
            k_ctx
        )
        hmm_dedo = candidatos[0]["dedo_candidato"]
        hist_opts = opciones_historicas_por_t(secuencias_historicas, idx_evento)

        for c in candidatos:
            tabla_detalle.append({
                "t": t,
                "midi": e["midi"],
                "ctx_key": e["ctx_key"],
                "dedo_anterior": dedo_anterior,
                "dedo_candidato": c["dedo_candidato"],
                "a_transicion": c["a_transicion"],
                "b_emision": c["b_emision"],
                "score": c["score"],
                "conteo_transicion": c["conteo_transicion"],
                "conteo_emision": c["conteo_emision"],
                "es_hmm_top": c["dedo_candidato"] == hmm_dedo,
            })

        if t in overrides:
            elegido = overrides[t]
            decision = "usuario"
            estado = "resuelto por usuario"
        else:
            if not hist_opts:
                elegido = hmm_dedo
                decision = "sin histórico exacto; usa HMM"
                estado = "automático"
            elif hmm_dedo in hist_opts:
                elegido = hmm_dedo
                decision = "HMM coincide con histórico"
                estado = "coincide"
            else:
                if len(hist_opts) == 1:
                    elegido = list(hist_opts.keys())[0]
                    decision = "histórico único aplicado"
                    estado = "corregido automático"
                else:
                    # Se detiene: pide elegir una de las opciones históricas o HMM
                    conflicto_pendiente = {
                        "t": t,
                        "midi": e["midi"],
                        "ctx_key": e["ctx_key"],
                        "hmm_dedo": hmm_dedo,
                        "historicas": hist_opts,
                        "candidatos": candidatos,
                    }
                    pasos.append({
                        "t": t,
                        "midi": e["midi"],
                        "hmm": hmm_dedo,
                        "historicas": hist_opts,
                        "elegido": None,
                        "decision": "conflicto: requiere usuario",
                        "estado": "pendiente"
                    })
                    break

        ruta.append(elegido)
        pasos.append({
            "t": t,
            "midi": e["midi"],
            "hmm": hmm_dedo,
            "historicas": hist_opts,
            "elegido": elegido,
            "decision": decision,
            "estado": estado
        })
        dedo_anterior = elegido

    return eventos, secuencias_historicas, pasos, tabla_detalle, ruta, conflicto_pendiente


HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>HMM asistido por históricos</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="https://cdn.datatables.net/2.0.8/css/dataTables.bootstrap5.css" rel="stylesheet">
<script>
window.MathJax = { tex: { inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] }, svg: { fontCache: 'global' } };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
body { background:#f4f6f8; }
.container-fluid { max-width:1650px; }
.card { border-radius:14px; box-shadow:0 3px 12px rgba(0,0,0,0.08); margin-bottom:24px; }
.formula-box { background:white; border-left:5px solid #0d6efd; padding:18px; font-size:18px; }
table.dataTable { font-size:13px; }
.ctx-small { font-size:12px; max-width:460px; white-space:normal; }
.score-ok { background:#d4edda !important; font-weight:bold; }
.pending { background:#fff3cd !important; }
.route { font-size: 1.8rem; font-weight: 700; }
</style>
</head>
<body>
<div class="container-fluid py-4">

<div class="card p-4">
<h1>HMM asistido por históricos</h1>
<p class="text-muted">
Esta demo no ejecuta Viterbi. Calcula una ruta local con $A_{ij} \\times B_j(o_t)$,
la contrasta contra digitaciones históricas exactas y se detiene cuando hay ambigüedad.
</p>
<div class="formula-box">
$$score_t(j)=A_{ij}\\times B_j(o_t)$$
</div>
</div>

<div class="card p-4">
<form method="POST">
<input type="hidden" name="overrides" value="{{ overrides_txt }}">
<div class="row">
    <div class="col-md-8">
        <label class="form-label">Melodía MIDI separada por comas</label>
        <input name="melodia" class="form-control" value="{{ melodia }}" {% if not melodia %}placeholder="Ej: 60,62,64,65,67,69"{% endif %}>
    </div>
    <div class="col-md-2">
        <label class="form-label">Dedo inicial manual</label>
        <select name="dedo_inicial" class="form-select">
            {% for d in [1,2,3,4,5] %}
            <option value="{{d}}" {% if d == dedo_inicial %}selected{% endif %}>Dedo {{d}}</option>
            {% endfor %}
        </select>
    </div>
    <div class="col-md-2 d-flex align-items-end gap-2">
        <button class="btn btn-primary w-100">Analizar</button>
        <button type="button" class="btn btn-outline-danger" onclick="window.location='/'">Reiniciar</button>
    </div>
</div>
</form>
</div>

{% if conflicto %}
<div class="card p-4 border-warning">
<h2>⚠️ Decisión asistida requerida</h2>
<p>
En <strong>t={{ conflicto.t }}</strong>, MIDI <strong>{{ conflicto.midi }}</strong>,
el HMM local propone <strong>dedo {{ conflicto.hmm_dedo }}</strong>, pero las digitaciones históricas exactas proponen otras opciones.
</p>
<p class="ctx-small"><code>{{ conflicto.ctx_key }}</code></p>

<form method="POST" class="d-flex flex-wrap gap-2">
<input type="hidden" name="melodia" value="{{ melodia }}">
<input type="hidden" name="dedo_inicial" value="{{ dedo_inicial }}">
<input type="hidden" name="overrides" value="{{ overrides_txt }}">
<input type="hidden" name="resolver_t" value="{{ conflicto.t }}">

<button name="resolver_dedo" value="{{ conflicto.hmm_dedo }}" class="btn btn-outline-primary">
Usar HMM: dedo {{ conflicto.hmm_dedo }}
</button>

{% for dedo, total in conflicto.historicas.items() %}
<button name="resolver_dedo" value="{{ dedo }}" class="btn btn-warning">
Usar histórico: dedo {{ dedo }} ({{ total }} coincidencia/s)
</button>
{% endfor %}
</form>
</div>
{% endif %}

{% if eventos %}
<div class="card p-4">
<h2>Ruta asistida actual</h2>
<p class="text-muted">Se calcula hasta el punto resuelto. Si hay conflicto pendiente, la ruta queda detenida allí.</p>
<div class="route">{{ ruta_txt }}</div>
</div>

<div class="card p-4">
<h2>Pasos de decisión</h2>
<table id="tabla_pasos" class="table table-striped table-bordered table-sm">
<thead>
<tr>
<th>t</th><th>MIDI</th><th>HMM local</th><th>Opciones históricas</th><th>Elegido</th><th>Decisión</th><th>Estado</th>
</tr>
</thead>
<tbody>
{% for p in pasos %}
<tr class="{% if p.estado == 'pendiente' %}pending{% endif %}">
<td>{{ p.t }}</td>
<td>{{ p.midi }}</td>
<td>{{ p.hmm }}</td>
<td>
{% if p.historicas %}
{% for dedo, total in p.historicas.items() %}
<span class="badge bg-secondary">{{ dedo }}: {{ total }}</span>
{% endfor %}
{% else %}
<span class="text-muted">sin histórico exacto</span>
{% endif %}
</td>
<td><strong>{{ p.elegido }}</strong></td>
<td>{{ p.decision }}</td>
<td>{{ p.estado }}</td>
</tr>
{% endfor %}
</tbody>
</table>
</div>

<div class="card p-4">
<h2>Detalle A × B calculado</h2>
<table id="tabla_detalle" class="table table-striped table-bordered table-sm">
<thead>
<tr>
<th>t</th><th>MIDI</th><th>ctx_key</th><th>Dedo anterior</th><th>Candidato</th><th>A</th><th>B</th><th>A × B</th><th>Conteo A</th><th>Conteo B</th><th>Top HMM</th>
</tr>
</thead>
<tbody>
{% for r in tabla_detalle %}
<tr class="{% if r.es_hmm_top %}score-ok{% endif %}">
<td>{{ r.t }}</td>
<td>{{ r.midi }}</td>
<td class="ctx-small"><code>{{ r.ctx_key }}</code></td>
<td>{{ r.dedo_anterior }}</td>
<td>{{ r.dedo_candidato }}</td>
<td>{{ "%.6f"|format(r.a_transicion) }}</td>
<td>{{ "%.6f"|format(r.b_emision) }}</td>
<td><strong>{{ "%.10f"|format(r.score) }}</strong></td>
<td>{{ r.conteo_transicion }}</td>
<td>{{ r.conteo_emision }}</td>
<td>{% if r.es_hmm_top %}✅{% endif %}</td>
</tr>
{% endfor %}
</tbody>
</table>
</div>

<div class="card p-4">
<h2>Coincidencias históricas exactas consideradas</h2>
{% if secuencias_historicas %}
<div class="accordion" id="accordionHistoricos">
{% for grupo in secuencias_historicas %}
<div class="accordion-item">
<h2 class="accordion-header" id="heading{{ loop.index }}">
<button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse{{ loop.index }}">
<strong>{{ grupo.dedos_historicos }}</strong>&nbsp;— {{ grupo.total }} coincidencia(s)
</button>
</h2>
<div id="collapse{{ loop.index }}" class="accordion-collapse collapse" data-bs-parent="#accordionHistoricos">
<div class="accordion-body">
<p><strong>Secuencia MIDI:</strong> {{ grupo.midis }}</p>
<table class="table table-sm table-bordered">
<thead><tr><th>Dataset ID</th><th>Dataset Name</th><th>Orden inicio</th><th>Orden fin</th></tr></thead>
<tbody>
{% for dataset in grupo.datasets %}
<tr><td>{{ dataset.dataset_id }}</td><td>{{ dataset.dataset_name }}</td><td>{{ dataset.order_inicio }}</td><td>{{ dataset.order_fin }}</td></tr>
{% endfor %}
</tbody>
</table>
</div>
</div>
</div>
{% endfor %}
</div>
{% else %}
<div class="alert alert-warning">No se encontró coincidencia histórica exacta con ese dedo inicial.</div>
{% endif %}
</div>
{% endif %}

</div>
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
<script src="https://cdn.datatables.net/2.0.8/js/dataTables.js"></script>
<script src="https://cdn.datatables.net/2.0.8/js/dataTables.bootstrap5.js"></script>
<script>
const config = { pageLength: 10, language: { url: 'https://cdn.datatables.net/plug-ins/2.0.8/i18n/es-ES.json' } };
if (document.querySelector('#tabla_pasos')) new DataTable('#tabla_pasos', config);
if (document.querySelector('#tabla_detalle')) new DataTable('#tabla_detalle', { ...config, pageLength: 25 });
</script>
</body>
</html>
"""


def registrar_scores(app):
    @app.route("/scores", methods=["GET", "POST"])
    def scores():
        # GET: estado limpio, sin resultados
        if request.method == "GET":
            return render_template_string(
                HTML,
                melodia="",
                dedo_inicial=1,
                overrides_txt="",
                eventos=[],
                pasos=[],
                tabla_detalle=[],
                ruta_txt="",
                conflicto=None,
                secuencias_historicas=[]
            )
        
        # POST: procesar el formulario
        melodia = request.form.get("melodia", "")
        if not melodia.strip():
            # Si no hay melodía, mostrar mensaje o volver con error
            return render_template_string(
                HTML,
                melodia="",
                dedo_inicial=1,
                overrides_txt="",
                eventos=[],
                pasos=[],
                tabla_detalle=[],
                ruta_txt="",
                conflicto=None,
                secuencias_historicas=[],
                error="Por favor ingrese una melodía"
            )
        
        dedo_inicial = int(request.form.get("dedo_inicial", 1))
        overrides = parse_override(request.form.get("overrides", ""))
        
        # Si el usuario resolvió un conflicto, agregamos esa decisión
        resolver_t = request.form.get("resolver_t")
        resolver_dedo = request.form.get("resolver_dedo")
        if resolver_t and resolver_dedo:
            overrides[int(resolver_t)] = int(resolver_dedo)
        
        midis = parse_melodia(melodia)
        if not midis:
            return render_template_string(
                HTML,
                melodia=melodia,
                dedo_inicial=dedo_inicial,
                overrides_txt="",
                eventos=[],
                pasos=[],
                tabla_detalle=[],
                ruta_txt="",
                conflicto=None,
                secuencias_historicas=[],
                error="La melodía debe contener al menos una nota"
            )
        
        eventos, secuencias_historicas, pasos, tabla_detalle, ruta, conflicto = ejecutar_asistido(midis, dedo_inicial, overrides)
        
        return render_template_string(
            HTML,
            melodia=melodia,
            dedo_inicial=dedo_inicial,
            overrides_txt=serialize_override(overrides),
            eventos=eventos,
            pasos=pasos,
            tabla_detalle=tabla_detalle,
            ruta_txt=" → ".join(map(str, ruta)) if ruta else "",
            conflicto=conflicto,
            secuencias_historicas=secuencias_historicas,
        )