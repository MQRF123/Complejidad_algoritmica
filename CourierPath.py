import reflex as rx
import pandas as pd
import plotly.express as px
import numpy as np
from math import radians, cos, sin, asin, sqrt
from typing import Tuple
from plotly.graph_objs import Figure
from reflex.vars import Var  


# --- Definición de URLs de los CSVs ---
URL_SUCURSALES = (
    "https://raw.githubusercontent.com/Complejidadalgorit/Datasets/refs/heads/main/sucursales.csv"
)
URL_DESTINOS = (
    "https://raw.githubusercontent.com/Complejidadalgorit/Datasets/refs/heads/main/destinos.csv"
)

# --- Función para leer CSVs (robusta) ---
def leer_datos_csv(ruta_csv: str) -> pd.DataFrame:
    try:
        return pd.read_csv(ruta_csv)
    except Exception as e:
        print(f"Error al leer {ruta_csv}: {e}")
        return pd.DataFrame()

# Cargar DataFrames
sucursales_df = leer_datos_csv(URL_SUCURSALES)
destinos_df   = leer_datos_csv(URL_DESTINOS)

# Verificar carga
if sucursales_df.empty or destinos_df.empty:
    raise ValueError("No se cargaron correctamente las sucursales o destinos.")

sucursales_df['latitud']  = sucursales_df['latitud'].astype(float)
sucursales_df['longitud'] = sucursales_df['longitud'].astype(float)
destinos_df['latitud']     = destinos_df['latitud'].astype(float)
destinos_df['longitud']    = destinos_df['longitud'].astype(float)

# Verificar carga
if sucursales_df.empty or destinos_df.empty:
    raise ValueError("No se cargaron correctamente las sucursales o destinos.")

# --- Función de Haversine para distancia en km ---
def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))

# --- (Opcional) Auxiliar para nodos ---
def haversine_from_nodes(a, b) -> float:
    # a = (lat, lon, name), b igual
    return haversine(a[1], a[0], b[1], b[0])

# --- Algoritmos: Held-Karp (TSP exacto) y Dijkstra ---
def held_karp(dist_matrix: np.ndarray) -> Tuple[list, float]:
    n = len(dist_matrix)
    C = {}
    # Inicializar
    for k in range(1, n):
        C[(1 << k, k)] = (dist_matrix[0, k], 0)
    # DP
    for size in range(2, n):
        for subset in [s for s in range(1<<n) if bin(s).count('1')==size and not (s & 1)]:
            for j in range(1, n):
                if not (subset & (1 << j)): continue
                prev = subset ^ (1 << j)
                costs = [(C[(prev, k)][0] + dist_matrix[k, j], k)
                         for k in range(1, n) if prev & (1 << k)]
                C[(subset, j)] = min(costs)
    # Cerrar ciclo
    full = (1 << n) - 1
    finals = [(C[(full ^ 1, k)][0] + dist_matrix[k, 0], k) for k in range(1, n)]
    opt, parent = min(finals)
    # Reconstruir ruta
    path = [0]
    mask, last = full ^ 1, parent
    for _ in range(n-1):
        path.append(last)
        _, prev = C[(mask, last)]
        mask ^= (1 << last)
        last = prev
    path.append(0)
    return path, opt

def dijkstra(dist_matrix: np.ndarray) -> list:
    n = dist_matrix.shape[0]
    dist = [float('inf')] * n
    dist[0] = 0
    visited = [False] * n
    for _ in range(n):
        u = min((d, i) for i, d in enumerate(dist) if not visited[i])[1]
        visited[u] = True
        for v in range(n):
            if not visited[v] and dist_matrix[u, v] > 0:
                alt = dist[u] + dist_matrix[u, v]
                if alt < dist[v]:
                    dist[v] = alt
    return dist

# --- Estado de la aplicación ---
class State(rx.State):
    sucursales = sucursales_df.to_dict('records')
    destinos   = destinos_df.to_dict('records')
    selected_sucursal = ""
    destinos_individuales: list[str] = [""] * 15
    algorithm = "TSP"
    route_coords: list = []
    segment_info: list = []
    total_distance = 0.0
    total_time = 0.0

    @rx.var
    def sucursal_options(self) -> list:
        return [s['nombre'] for s in self.sucursales]

    @rx.var
    def destino_options(self) -> list:
        return [d['direccion'] for d in self.destinos]

    @rx.var
    def algorithm_options(self) -> list:
        return ["TSP", "Dijkstra"]
    
    def set_individual_destino(self, index: int, value: str):
        self.destinos_individuales[index] = value

    @rx.var
    def summary(self) -> str:
        return f"Distancia total: {self.total_distance:.2f} km | Tiempo: {self.total_time:.0f} min"

    def reset_state(self):
        self.selected_sucursal = ""
        self.destinos_individuales = [""] * 15
        self.algorithm = "TSP"
        self.route_coords = []
        self.segment_info = []
        self.total_distance = 0.0
        self.total_time = 0.0

    def set_selected_sucursal(self, nombre: str):
        self.selected_sucursal = nombre
        self.destinos_individuales = [""] * 15
        self.route_coords = []
        self.segment_info = []

    def set_selected_destinos(self, ds):
        if isinstance(ds, str):
            ds = [ds]
        elif not isinstance(ds, list):
            ds = list(ds)
        self.selected_destinos = ds[:15]
        self.route_coords = []
        self.segment_info = []

    def set_algorithm(self, alg: str):
        self.algorithm = alg
        self.route_coords = []
        self.segment_info = []

    def calcular_ruta(self):
        if not self.selected_sucursal:
            return

        valid_destinos = [d for d in self.destinos_individuales if d]
        sucursal = next((s for s in self.sucursales if s['nombre'] == self.selected_sucursal), None)
        if not sucursal or not valid_destinos:
            return

        # Construir nodos: (lat, lon, nombre)
        nodos = [(sucursal['latitud'], sucursal['longitud'], sucursal['nombre'])]
        for dname in valid_destinos:
            d = next((d for d in self.destinos if d['direccion'] == dname), None)
            if d:
                nodos.append((d['latitud'], d['longitud'], d['direccion']))

        n = len(nodos)
        if n < 2:
            return

        # Construir matriz de distancias
        dm = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # llamada CORRECTA a haversine
                    dm[i, j] = haversine(nodos[i][1], nodos[i][0], nodos[j][1], nodos[j][0])

        coords = []
        info = []
        dist_total = 0.0
        tiempo_total = 0.0

        if self.algorithm == "TSP":
            if n > 12:
                info.append("⚠️ Held-Karp es demasiado lento para más de 12 puntos. Usa otro algoritmo.")
            else:
                path, dist_total = held_karp(dm)
                for a, b in zip(path[:-1], path[1:]):
                    dseg = dm[a, b]
                    tseg = dseg / 40 * 60
                    info.append(f"{nodos[a][2]} → {nodos[b][2]}: {dseg:.2f} km, {tseg:.0f} min")
                    coords.append((nodos[a][0], nodos[a][1]))
                    tiempo_total += tseg
                coords.append((nodos[path[-1]][0], nodos[path[-1]][1]))

        elif self.algorithm == "Dijkstra":
            dists = dijkstra(dm)
            coords.append((nodos[0][0], nodos[0][1]))
            for i in range(1, n):
                dseg = dists[i]
                tseg = dseg / 40 * 60
                info.append(f"{nodos[0][2]} → {nodos[i][2]}: {dseg:.2f} km, {tseg:.0f} min")
                coords.append((nodos[i][0], nodos[i][1]))
                dist_total += dseg
                tiempo_total += tseg

        else:  # TCP - Greedy vecino más cercano
            visitados = [False] * n
            visitados[0] = True
            actual = 0
            coords.append((nodos[0][0], nodos[0][1]))

            for _ in range(1, n):
                sig = -1
                min_dist = float("inf")
                for i in range(1, n):
                    if not visitados[i] and dm[actual][i] < min_dist:
                        min_dist = dm[actual][i]
                        sig = i
                if sig == -1:
                    break
                visitados[sig] = True
                t = min_dist / 40 * 60
                tiempo_total += t
                dist_total += min_dist
                info.append(f"{nodos[actual][2]} → {nodos[sig][2]}: {min_dist:.2f} km, {t:.0f} min")
                coords.append((nodos[sig][0], nodos[sig][1]))
                actual = sig

            # regreso a sucursal
            retorno = dm[actual][0]
            t = retorno / 40 * 60
            tiempo_total += t
            dist_total += retorno
            info.append(f"{nodos[actual][2]} → {nodos[0][2]}: {retorno:.2f} km, {t:.0f} min")
            coords.append((nodos[0][0], nodos[0][1]))

        self.route_coords    = coords
        self.segment_info    = info
        self.total_distance  = dist_total
        self.total_time      = tiempo_total

    @rx.var
    def updated_map_figure(self) -> Figure:
        if not self.route_coords:
            return px.scatter_mapbox(
                sucursales_df,
                lat='latitud', lon='longitud', hover_name='nombre',
                mapbox_style='open-street-map', zoom=11
            )
        dfp = pd.DataFrame(self.route_coords, columns=['lat', 'lon'])
        fig = px.scatter_mapbox(
            dfp, lat='lat', lon='lon', mapbox_style='open-street-map', zoom=11
        )
        fig.add_trace(px.line_mapbox(lat=dfp['lat'], lon=dfp['lon']).data[0])
        return fig

# --- Estilos globales ---
def get_global_style():
    return {"body": {"backgroundColor": "white", "color": "black", "fontFamily": "Arial, sans-serif"}}

# --- Layout de la página ---
def index() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.select(
                State.algorithm_options,
                placeholder="Algoritmo",
                on_change=State.set_algorithm,
                width='25%', searchable=True
            ),
            rx.select(
                State.sucursal_options,
                placeholder="Sucursal",
                on_change=State.set_selected_sucursal,
                width='35%', searchable=True
            )
        ),
        rx.vstack(
            *[
                rx.select(
                    State.destino_options,
                    placeholder=f"Destino {i+1}",
                    on_change=lambda value, i=i: State.set_individual_destino(i, value),
                    searchable=True,
                    allow_deselect=True,
                    width='100%'
                )
                for i in range(15)
            ],
            spacing='1'
        ),
        rx.button("Calcular Ruta", on_click=State.calcular_ruta, color_scheme='blue', margin_top='1rem'),
        rx.hstack(
            rx.box(rx.foreach(State.segment_info, lambda s: rx.text(s)), border='1px solid gray', padding='1rem', width='50%'),
            rx.box(rx.plotly(data=State.updated_map_figure, layout={'height': 500}), width='50%')
        ),
        rx.text(State.summary, size='5'),
        padding='2rem'
    )

# --- Creación y ejecución de la app ---
app = rx.App()
app.add_page(index, on_load=State.reset_state)
