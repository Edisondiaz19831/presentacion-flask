from flask import Flask, render_template, send_from_directory, request, jsonify, session
from pathlib import Path
import sqlite3
import pandas as pd
import traceback
import os
import re

app = Flask(__name__)

app.secret_key = os.environ.get("SECRET_KEY", "clave-secreta-por-defecto")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "markov.db"

DEDOS = [1, 2, 3, 4, 5]
EPSILON = 0.001


# ==================== CONEXIÓN SQLITE ====================

def conectar():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"No existe {DB_PATH.name}. Verifica que markov.db esté en la raíz del proyecto."
        )

    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def conectar_db():
    conn = conectar()
    conn.row_factory = sqlite3.Row
    return conn


def nombre_tabla_seguro(tabla):
    return re.match(r"^[A-Za-z0-9_]+$", tabla) is not None


# ==================== CORS ====================

@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response


# ==================== RUTAS PRESENTACIÓN ====================

@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/plano_piano")
def plano_piano():
    return render_template("plano_piano.html")


@app.route("/markov_dedos")
def markov_dedos():
    return render_template("markov_dedos.html")


@app.route("/prior")
def prior():
    return render_template("hmm_prior_inicial.html")


@app.route("/prueba_modelo")
def prueba_modelo():
    return render_template("prueba_modelo.html")


# ==================== ARCHIVOS ESTÁTICOS ====================

@app.route("/css/<path:filename>")
def css(filename):
    return send_from_directory(BASE_DIR / "css", filename)


@app.route("/js/<path:filename>")
def js(filename):
    return send_from_directory(BASE_DIR / "js", filename)


@app.route("/img/<path:filename>")
def img(filename):
    return send_from_directory(BASE_DIR / "img", filename)


@app.route("/fonts/<path:filename>")
def fonts(filename):
    return send_from_directory(BASE_DIR / "fonts", filename)


# ==================== FUNCIONES DEL MODELO ====================

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


def calcular_run(size_prev, size_next, dir_prev, dir_next, is_first, is_last):
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


def construir_ctx_key(midi, prev_midi, next_midi, is_first, is_last):
    delta_prev = midi - prev_midi if prev_midi is not None else None
    delta_next = next_midi - midi if next_midi is not None else None

    dir_prev = direccion(delta_prev)
    dir_next = direccion(delta_next)

    size_prev = size_salto(delta_prev)
    size_next = size_salto(delta_next)

    mov_prev = movimiento(delta_prev)
    mov_next = movimiento(delta_next)

    black = 1 if es_tecla_negra(midi) else 0

    run = calcular_run(
        size_prev,
        size_next,
        dir_prev,
        dir_next,
        is_first,
        is_last
    )

    return (
        f"mp={mov_prev}|mn={mov_next}|dir_prev={dir_prev}|dir_next={dir_next}"
        f"|size_prev={size_prev}|size_next={size_next}|black={black}|run={run}"
    )


def construir_contextos(secuencia):
    eventos = []

    for i, midi in enumerate(secuencia):
        prev_midi = secuencia[i - 1] if i > 0 else None
        next_midi = secuencia[i + 1] if i < len(secuencia) - 1 else None

        is_first = i == 0
        is_last = i == len(secuencia) - 1

        ctx_key = construir_ctx_key(
            midi,
            prev_midi,
            next_midi,
            is_first,
            is_last
        )

        eventos.append({
            "t": i + 1,
            "midi": midi,
            "ctx_key": ctx_key
        })

    return eventos


def cargar_transiciones(conn):
    try:
        df = pd.read_sql_query("""
            SELECT prev_finger, finger, COUNT(*) AS total
            FROM dataset_event
            WHERE prev_finger IS NOT NULL
              AND finger IS NOT NULL
              AND prev_finger IN (1,2,3,4,5)
              AND finger IN (1,2,3,4,5)
            GROUP BY prev_finger, finger
        """, conn)

        conteos = df.pivot(
            index="prev_finger",
            columns="finger",
            values="total"
        ).fillna(0)

        conteos = conteos.reindex(
            index=DEDOS,
            columns=DEDOS,
            fill_value=0
        )

        A = (conteos + EPSILON).div(
            conteos.sum(axis=1) + len(DEDOS) * EPSILON,
            axis=0
        )

        return A

    except Exception as e:
        print(f"Error cargando transiciones: {e}")

        import numpy as np

        return pd.DataFrame(
            np.ones((len(DEDOS), len(DEDOS))) / len(DEDOS),
            index=DEDOS,
            columns=DEDOS
        )


def cargar_emisiones(conn):
    try:
        df = pd.read_sql_query("""
            SELECT finger, ctx_key, COUNT(*) AS total
            FROM dataset_event
            WHERE finger IS NOT NULL
              AND ctx_key IS NOT NULL
              AND finger IN (1,2,3,4,5)
            GROUP BY finger, ctx_key
        """, conn)

        conteos = df.pivot(
            index="finger",
            columns="ctx_key",
            values="total"
        ).fillna(0)

        conteos = conteos.reindex(index=DEDOS, fill_value=0)

        k = max(conteos.shape[1], 1)

        B = (conteos + EPSILON).div(
            conteos.sum(axis=1) + k * EPSILON,
            axis=0
        )

        return B

    except Exception as e:
        print(f"Error cargando emisiones: {e}")
        return pd.DataFrame(index=DEDOS)


def cargar_historicos(conn):
    try:
        return pd.read_sql_query("""
            SELECT dataset_id, "order", midi, finger, ctx_key
            FROM dataset_event
            WHERE finger IS NOT NULL
              AND finger IN (1,2,3,4,5)
              AND ctx_key IS NOT NULL
            ORDER BY dataset_id, "order"
        """, conn)

    except Exception as e:
        print(f"Error cargando históricos: {e}")
        return pd.DataFrame()


def opciones_historicas_por_ventana(df_hist, secuencia, ruta, i):
    opciones = []

    midis_previos = secuencia[:i]
    dedos_previos = ruta[:]
    n = len(midis_previos)

    if n == 0 or df_hist.empty:
        return pd.DataFrame(), pd.DataFrame()

    try:
        for dataset_id, grupo in df_hist.groupby("dataset_id"):
            grupo = grupo.sort_values("order").reset_index(drop=True)

            if len(grupo) <= n:
                continue

            midis_hist = grupo["midi"].astype(int).tolist()
            dedos_hist = grupo["finger"].astype(int).tolist()
            ordenes_hist = grupo["order"].tolist()

            for inicio in range(0, len(grupo) - n):
                ventana_midis = midis_hist[inicio:inicio + n]
                ventana_dedos = dedos_hist[inicio:inicio + n]

                if ventana_midis == midis_previos and ventana_dedos == dedos_previos:
                    if inicio + n < len(midis_hist):
                        opciones.append({
                            "dataset_id": dataset_id,
                            "order_inicio": ordenes_hist[inicio],
                            "order_fin": ordenes_hist[inicio + n - 1],
                            "midi_actual_historico": midis_hist[inicio + n],
                            "dedo_actual_historico": dedos_hist[inicio + n],
                        })

        if not opciones:
            return pd.DataFrame(), pd.DataFrame()

        detalle = pd.DataFrame(opciones)

        resumen = (
            detalle.groupby("dedo_actual_historico")
            .size()
            .reset_index(name="veces")
            .rename(columns={"dedo_actual_historico": "siguiente_dedo"})
            .sort_values("veces", ascending=False)
        )

        return resumen, detalle

    except Exception as e:
        print(f"Error en opciones_historicas: {e}")
        return pd.DataFrame(), pd.DataFrame()


# ==================== API DEL MODELO ====================

@app.route("/iniciar", methods=["POST"])
def iniciar():
    try:
        data = request.json

        dedo_inicial = int(data["dedo_inicial"])
        secuencia = [
            int(x.strip())
            for x in data["secuencia"].split(",")
            if x.strip()
        ]

        if not secuencia:
            return jsonify({"error": "La secuencia no puede estar vacía"}), 400

        if dedo_inicial not in DEDOS:
            return jsonify({"error": f"Dedo inicial debe ser {DEDOS}"}), 400

        conn = conectar()
        cargar_transiciones(conn)
        cargar_emisiones(conn)
        cargar_historicos(conn)
        conn.close()

        eventos = construir_contextos(secuencia)

        session["secuencia"] = secuencia
        session["dedo_inicial"] = dedo_inicial
        session["ruta"] = [dedo_inicial]
        session["dedo_anterior"] = dedo_inicial
        session["paso_actual"] = 0
        session["eventos"] = eventos
        session["total_pasos"] = len(eventos)

        return jsonify({
            "eventos": eventos,
            "ruta": session["ruta"],
            "total_pasos": len(eventos)
        })

    except Exception as e:
        print(f"Error en iniciar: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/siguiente_paso", methods=["POST"])
def siguiente_paso():
    try:
        if "secuencia" not in session:
            return jsonify({"error": "Sesión no iniciada"}), 400

        secuencia = session.get("secuencia")
        ruta = session.get("ruta", [])
        dedo_anterior = session.get("dedo_anterior")
        paso_actual = session.get("paso_actual", 0)
        eventos = session.get("eventos", [])

        if not eventos:
            return jsonify({"error": "No hay eventos en sesión"}), 400

        if paso_actual >= len(eventos) - 1:
            return jsonify({
                "completado": True,
                "ruta": ruta,
                "mensaje": "Predicción completada"
            })

        next_paso = paso_actual + 1
        e = eventos[next_paso]
        ctx = e.get("ctx_key", "")

        conn = conectar()
        A = cargar_transiciones(conn)
        B = cargar_emisiones(conn)
        df_hist = cargar_historicos(conn)
        conn.close()

        candidatos = []

        for dedo_candidato in DEDOS:
            if dedo_anterior in A.index and dedo_candidato in A.columns:
                a = A.loc[dedo_anterior, dedo_candidato]
            else:
                a = EPSILON

            if ctx in B.columns and dedo_candidato in B.index:
                b = B.loc[dedo_candidato, ctx]
            else:
                b = EPSILON

            score = a * b

            candidatos.append({
                "dedo": dedo_candidato,
                "A": round(float(a), 4),
                "B": round(float(b), 4),
                "score": round(float(score), 4)
            })

        df_candidatos = pd.DataFrame(candidatos).sort_values(
            "score",
            ascending=False
        )

        dedo_modelo = int(df_candidatos.iloc[0]["dedo"])

        opciones_hist, detalle_hist = opciones_historicas_por_ventana(
            df_hist,
            secuencia,
            ruta,
            next_paso
        )

        necesita_interaccion = False
        mensaje = ""
        opciones_disponibles = []
        dedo_recomendado = None

        if not opciones_hist.empty:
            dedos_validos = opciones_hist["siguiente_dedo"].tolist()

            if len(dedos_validos) == 1:
                unico_historico = int(dedos_validos[0])
                dedo_recomendado = unico_historico

                if unico_historico == dedo_modelo:
                    mensaje = (
                        f"✅ Histórico único dedo {unico_historico} "
                        "y coincide con Markov."
                    )
                else:
                    mensaje = (
                        f"✅ Histórico único dedo {unico_historico}; "
                        "se prioriza histórico."
                    )
            else:
                necesita_interaccion = True
                opciones_disponibles = [int(d) for d in dedos_validos]
                mensaje = "📚 Hay múltiples opciones históricas. Elige un dedo."
        else:
            dedo_recomendado = dedo_modelo
            mensaje = f"🤖 No hay continuación histórica. Se usa Markov: dedo {dedo_modelo}."

        if not necesita_interaccion and dedo_recomendado is not None:
            ruta.append(dedo_recomendado)

            session["ruta"] = ruta
            session["dedo_anterior"] = dedo_recomendado
            session["paso_actual"] = next_paso

            completado = next_paso >= len(eventos) - 1

            return jsonify({
                "necesita_interaccion": False,
                "dedo_elegido": dedo_recomendado,
                "mensaje": mensaje,
                "ruta": ruta,
                "completado": completado,
                "paso_actual": next_paso
            })

        return jsonify({
            "necesita_interaccion": True,
            "mensaje": mensaje,
            "candidatos": df_candidatos.head(5).to_dict("records"),
            "opciones_historicas": opciones_hist.to_dict("records"),
            "detalle_historico": detalle_hist.head(10).to_dict("records"),
            "dedo_modelo": dedo_modelo,
            "opciones_disponibles": opciones_disponibles,
            "paso_actual": next_paso,
            "midi_actual": e.get("midi", ""),
            "ctx": ctx
        })

    except Exception as e:
        print(f"Error en siguiente_paso: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/elegir_dedo", methods=["POST"])
def elegir_dedo():
    try:
        data = request.json
        dedo_elegido = int(data["dedo_elegido"])

        ruta = session.get("ruta", [])
        paso_actual = session.get("paso_actual", 0)
        eventos = session.get("eventos", [])

        if dedo_elegido not in DEDOS:
            return jsonify({"error": f"Dedo inválido. Debe ser {DEDOS}"}), 400

        ruta.append(dedo_elegido)

        session["ruta"] = ruta
        session["dedo_anterior"] = dedo_elegido
        session["paso_actual"] = paso_actual + 1

        completado = paso_actual + 1 >= len(eventos) - 1

        return jsonify({
            "dedo_elegido": dedo_elegido,
            "ruta": ruta,
            "completado": completado,
            "paso_actual": paso_actual + 1
        })

    except Exception as e:
        print(f"Error en elegir_dedo: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ==================== VISUALIZADOR SQLITE ====================

@app.route("/visualizar_sqlite")
def visualizar_sqlite():
    try:
        conn = conectar_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            AND name NOT LIKE 'sqlite_%';
        """)

        tablas = [row["name"] for row in cursor.fetchall()]

        datos = []
        columnas = []
        nombre_tabla = None
        total_filas = 0

        if tablas:
            nombre_tabla = tablas[0]

            if nombre_tabla_seguro(nombre_tabla):
                cursor.execute(f'SELECT * FROM "{nombre_tabla}" LIMIT 1000;')
                datos = cursor.fetchall()

                if datos:
                    columnas = list(datos[0].keys())

                cursor.execute(f'SELECT COUNT(*) FROM "{nombre_tabla}";')
                total_filas = cursor.fetchone()[0]

        conn.close()

        return render_template(
            "sqlite_viewer.html",
            tablas=tablas,
            datos=datos,
            columnas=columnas,
            nombre_tabla=nombre_tabla,
            total_filas=total_filas
        )

    except Exception as e:
        return render_template(
            "sqlite_viewer.html",
            error=f"Error al leer la base de datos: {str(e)}"
        )


@app.route("/api/tablas")
def api_tablas():
    try:
        conn = conectar_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            AND name NOT LIKE 'sqlite_%';
        """)

        tablas = [row["name"] for row in cursor.fetchall()]
        conn.close()

        return jsonify({"tablas": tablas})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/datos/<tabla>")
def api_datos_tabla(tabla):
    try:
        if not nombre_tabla_seguro(tabla):
            return jsonify({"error": "Nombre de tabla no válido"}), 400

        conn = conectar_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            AND name = ?;
        """, (tabla,))

        if not cursor.fetchone():
            conn.close()
            return jsonify({"error": f'La tabla "{tabla}" no existe'}), 404

        cursor.execute(f'SELECT * FROM "{tabla}" LIMIT 1000;')
        datos = [dict(row) for row in cursor.fetchall()]

        cursor.execute(f'SELECT COUNT(*) FROM "{tabla}";')
        total = cursor.fetchone()[0]

        conn.close()

        return jsonify({
            "tabla": tabla,
            "datos": datos,
            "mostrados": len(datos),
            "total": total
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/info")
def api_info():
    try:
        conn = conectar_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            AND name NOT LIKE 'sqlite_%';
        """)

        tablas = [row["name"] for row in cursor.fetchall()]
        info_tablas = []

        for tabla in tablas:
            if not nombre_tabla_seguro(tabla):
                continue

            cursor.execute(f'SELECT COUNT(*) FROM "{tabla}";')
            total_filas = cursor.fetchone()[0]

            cursor.execute(f'PRAGMA table_info("{tabla}");')
            columnas = [col[1] for col in cursor.fetchall()]

            info_tablas.append({
                "nombre": tabla,
                "filas": total_filas,
                "columnas": columnas,
                "total_columnas": len(columnas)
            })

        conn.close()

        return jsonify({
            "base_datos": DB_PATH.name,
            "existe": DB_PATH.exists(),
            "tablas": info_tablas,
            "total_tablas": len(info_tablas)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "database_exists": DB_PATH.exists(),
        "database": DB_PATH.name
    })


# ==================== EJECUCIÓN LOCAL ====================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )