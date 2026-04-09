"""
========================================================================
SIMULACIÓN DE CONTROL PID CON MESA (Python) — v3.0
Controlador PID para regulación de velocidad de un agente móvil
========================================================================
Novedades v3.0:
    [v3 - FIX 1]  Simulación indefinida: el bucle corre hasta que el
                  usuario cierra la ventana. No hay límite de tiempo.
    [v3 - FIX 2]  Perturbaciones interactivas en tiempo real:
                  · Teclado  →  teclas del teclado (foco en la ventana)
                  · Botones  →  botones en pantalla con matplotlib.widgets

    Teclas / botones disponibles:
        [↑] o botón "＋ Setpoint"   → sube el setpoint  +5 m/s
        [↓] o botón "－ Setpoint"   → baja el setpoint  −5 m/s
        [A] o botón "Kick ＋"       → golpe de velocidad +8 m/s al agente
        [Z] o botón "Kick －"       → golpe de velocidad −8 m/s al agente
        [R] o botón "Reset"         → reinicia velocidades y estado PID
        [Q] o cerrar ventana        → termina la simulación

    Cada perturbación queda marcada en la gráfica con una línea
    vertical anotada, para análisis académico.

Mejoras heredadas de v2.0:
    [MEJORA 1]  Anti-windup real sobre el acumulador integral.
    [MEJORA 2]  DataCollector nativo de MESA.
    [MEJORA 3]  Filtro EMA en el término derivativo.
========================================================================
"""

# ── Importaciones ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("TkAgg")          # cambia a "Qt5Agg" si TkAgg no está disponible

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
import numpy as np
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector


# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTES DE PERTURBACIÓN
# ════════════════════════════════════════════════════════════════════════════
SETPOINT_STEP = 5.0    # cambio de setpoint por pulsación [m/s]
KICK_DELTA    = 8.0    # impulso de velocidad aplicado al agente [m/s]
SETPOINT_MIN  = 0.0    # setpoint mínimo permitido [m/s]
SETPOINT_MAX  = 60.0   # setpoint máximo permitido [m/s]
WINDOW_SIZE   = 300    # pasos visibles en pantalla (ventana deslizante)


# ════════════════════════════════════════════════════════════════════════════
#  FUNCIONES AUXILIARES PARA EL DATACOLLECTOR
# ════════════════════════════════════════════════════════════════════════════
def get_velocity(a)    -> float: return a.velocity
def get_error(a)       -> float: return (a.model.setpoint - a.velocity) if a.use_pid else float("nan")
def get_integral(a)    -> float: return a._integral
def get_d_filtered(a)  -> float: return a._d_filtered
def get_acceleration(a)-> float: return a.acceleration


# ════════════════════════════════════════════════════════════════════════════
#  1. AGENTE PID
# ════════════════════════════════════════════════════════════════════════════
class VehicleAgent(Agent):
    """
    Agente robot/vehículo con controlador PID completo.
    Incluye anti-windup [M1] y filtro EMA en la derivada [M3].
    """

    def __init__(self, unique_id, model, v_init=0.0, use_pid=True):
        super().__init__(unique_id, model)
        self.velocity      = v_init
        self.acceleration  = 0.0
        self.use_pid       = use_pid

        self.Kp    = model.Kp
        self.Ki    = model.Ki
        self.Kd    = model.Kd
        self.dt    = model.dt
        self.alpha = model.alpha

        self._u_max        = model.u_max
        self._integral     = 0.0
        self._integral_max = model.u_max / model.Ki if model.Ki != 0 else 1e9
        self._prev_error   = 0.0
        self._d_filtered   = 0.0

        self.damping   = model.damping
        self.noise_std = model.noise_std

    def reset_pid_state(self):
        """Reinicia los estados internos del PID (útil tras reset interactivo)."""
        self._integral    = 0.0
        self._prev_error  = 0.0
        self._d_filtered  = 0.0

    def _pid_output(self, setpoint: float) -> float:
        """Calcula u(k) con anti-windup [M1] y derivada filtrada [M3]."""
        v_meas = self.velocity + (
            np.random.normal(0.0, self.noise_std) if self.noise_std > 0 else 0.0
        )
        error = setpoint - v_meas

        # Proporcional
        P = self.Kp * error

        # Integral con anti-windup [M1]: se satura el acumulador, no la salida
        self._integral += error * self.dt
        self._integral  = np.clip(self._integral,
                                  -self._integral_max, self._integral_max)
        I = self.Ki * self._integral

        # Derivada filtrada EMA [M3]
        d_raw = (error - self._prev_error) / self.dt
        self._d_filtered = self.alpha * d_raw + (1.0 - self.alpha) * self._d_filtered
        D = self.Kd * self._d_filtered
        self._prev_error = error

        return float(np.clip(P + I + D, -self._u_max, self._u_max))

    def step(self):
        """Un tick de simulación: calcular aceleración y actualizar velocidad."""
        if self.use_pid:
            self.acceleration = self._pid_output(self.model.setpoint)
        else:
            self.acceleration = 2.0  # lazo abierto: aceleración fija

        v_dot = self.acceleration - self.damping * self.velocity
        self.velocity = max(0.0, self.velocity + v_dot * self.dt)


# ════════════════════════════════════════════════════════════════════════════
#  2. MODELO MESA
# ════════════════════════════════════════════════════════════════════════════
class VehicleModel(Model):
    """Modelo MESA con DataCollector [M2] y soporte de perturbaciones."""

    def __init__(self, Kp=5.0, Ki=1.2, Kd=0.8, dt=0.05,
                 setpoint=20.0, damping=0.3, v_init=0.0,
                 u_max=50.0, alpha=0.3, noise_std=0.4):
        super().__init__()
        self.Kp = Kp; self.Ki = Ki; self.Kd = Kd
        self.dt = dt; self.setpoint = setpoint
        self.damping = damping; self.u_max = u_max
        self.alpha = alpha; self.noise_std = noise_std

        self.schedule     = BaseScheduler(self)
        self.current_time = 0.0
        self.step_count   = 0

        self.schedule.add(VehicleAgent(0, self, v_init=v_init, use_pid=True))
        self.schedule.add(VehicleAgent(1, self, v_init=v_init, use_pid=False))

        self.datacollector = DataCollector(agent_reporters={
            "Velocity"    : get_velocity,
            "Acceleration": get_acceleration,
            "Error"       : get_error,
            "Integral"    : get_integral,
            "D_filtered"  : get_d_filtered,
        })

        # ── Registro de perturbaciones para marcar en la gráfica ─────────
        # Cada entrada: (step_number, tipo_str, color_str)
        self.disturbances: list[tuple[int, str, str]] = []

    def step(self):
        self.schedule.step()
        self.current_time += self.dt
        self.step_count   += 1
        self.datacollector.collect(self)

    # ── API de perturbaciones ────────────────────────────────────────────────
    def apply_setpoint_up(self):
        """Sube el setpoint SETPOINT_STEP m/s (simula cambio de referencia)."""
        self.setpoint = min(SETPOINT_MAX, self.setpoint + SETPOINT_STEP)
        self.disturbances.append((self.step_count, f"SP→{self.setpoint:.0f}", "#f0883e"))
        print(f"  [↑] Setpoint → {self.setpoint:.1f} m/s  (t={self.current_time:.2f}s)")

    def apply_setpoint_down(self):
        """Baja el setpoint SETPOINT_STEP m/s."""
        self.setpoint = max(SETPOINT_MIN, self.setpoint - SETPOINT_STEP)
        self.disturbances.append((self.step_count, f"SP→{self.setpoint:.0f}", "#f0883e"))
        print(f"  [↓] Setpoint → {self.setpoint:.1f} m/s  (t={self.current_time:.2f}s)")

    def apply_kick_positive(self):
        """
        Perturbación positiva: añade KICK_DELTA m/s a la velocidad del agente PID.
        Simula un empuje externo (viento, rampa, colisión suave).
        """
        pid_agent = self.schedule.agents[0]
        pid_agent.velocity += KICK_DELTA
        self.disturbances.append((self.step_count, f"+{KICK_DELTA:.0f}m/s", "#3fb950"))
        print(f"  [A] Kick +{KICK_DELTA} m/s  → v={pid_agent.velocity:.2f}  (t={self.current_time:.2f}s)")

    def apply_kick_negative(self):
        """
        Perturbación negativa: resta KICK_DELTA m/s a la velocidad.
        Simula frenado brusco, subida, resistencia repentina.
        """
        pid_agent = self.schedule.agents[0]
        pid_agent.velocity = max(0.0, pid_agent.velocity - KICK_DELTA)
        self.disturbances.append((self.step_count, f"−{KICK_DELTA:.0f}m/s", "#f85149"))
        print(f"  [Z] Kick −{KICK_DELTA} m/s  → v={pid_agent.velocity:.2f}  (t={self.current_time:.2f}s)")

    def apply_reset(self):
        """Reinicia velocidades y estado interno del PID a condiciones iniciales."""
        for agent in self.schedule.agents:
            agent.velocity = 0.0
            agent.acceleration = 0.0
            if agent.use_pid:
                agent.reset_pid_state()
        self.disturbances.append((self.step_count, "RESET", "#d2a8ff"))
        print(f"  [R] Reset  (t={self.current_time:.2f}s)")

    # ── Helpers de acceso al DataFrame del DataCollector [M2] ────────────────
    def _series(self, aid: int, col: str) -> np.ndarray:
        df = self.datacollector.get_agent_vars_dataframe()
        return df.xs(aid, level="AgentID")[col].to_numpy()

    @property
    def time_data(self)      -> np.ndarray:
        n = len(self.datacollector.get_agent_vars_dataframe().xs(0, level="AgentID"))
        return np.arange(1, n + 1) * self.dt
    @property
    def vel_pid_data(self)   -> np.ndarray: return self._series(0, "Velocity")
    @property
    def vel_open_data(self)  -> np.ndarray: return self._series(1, "Velocity")
    @property
    def error_pid_data(self) -> np.ndarray: return self._series(0, "Error")
    @property
    def integral_data(self)  -> np.ndarray: return self._series(0, "Integral")
    @property
    def d_filtered_data(self)-> np.ndarray: return self._series(0, "D_filtered")


# ════════════════════════════════════════════════════════════════════════════
#  3. VISUALIZADOR INTERACTIVO EN TIEMPO REAL (v3.0)
# ════════════════════════════════════════════════════════════════════════════
class RealTimePlotter:
    """
    Visualizador con:
        · 4 gráficas de tiempo real con ventana deslizante.
        · 5 botones matplotlib.widgets.Button en pantalla.
        · Captura de teclado con fig.canvas.mpl_connect('key_press_event').
        · Líneas verticales automáticas en cada perturbación.

    Cómo funciona la interacción:
    ──────────────────────────────
    Teclado: mpl_connect registra un callback on_key() que se activa
             cada vez que el usuario pulsa una tecla con la ventana
             enfocada. El callback setea un flag en self.pending_action,
             que el bucle principal consume en el siguiente tick.

    Botones: Button de matplotlib.widgets llama directamente al mismo
             callback, garantizando que ambas interfaces sean equivalentes.

    Perturbaciones: cada acción registra (step, etiqueta, color) en
             model.disturbances. update() dibuja axvline en cada gráfica.
    """

    _CB = "#161b22"   # fondo paneles
    _CS = "#30363d"   # color bordes
    _CT = "#8b949e"   # color ticks / labels

    def __init__(self, model: VehicleModel):
        self.model   = model
        self.running = True          # False cuando el usuario cierra la ventana

        # ── Acción pendiente del teclado (consumida en el bucle) ──────────
        self.pending_action: str | None = None

        # ── Líneas de perturbación ya dibujadas (para no redibujar) ───────
        self._drawn_disturbances: set[int] = set()

        # ── Figura ────────────────────────────────────────────────────────
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(15, 9), facecolor="#0d1117")
        self.fig.suptitle(
            "MESA PID v3.0  ·  Simulación Indefinida  ·  Perturbaciones Interactivas",
            fontsize=12, fontweight="bold", color="#e6edf3", y=0.99
        )

        # ── Layout: gráficas arriba, botones abajo ────────────────────────
        # Reservamos 0.13 de alto para los botones
        gs = gridspec.GridSpec(2, 3, figure=self.fig,
                               top=0.93, bottom=0.18,
                               hspace=0.50, wspace=0.38)

        def _sty(ax, title, ylabel):
            ax.set_facecolor(self._CB)
            ax.set_title(title, color="#58a6ff", fontsize=9.5)
            ax.set_xlabel("Tiempo [s]", color=self._CT, fontsize=8)
            ax.set_ylabel(ylabel,       color=self._CT, fontsize=8)
            ax.tick_params(colors=self._CT)
            for sp in ax.spines.values(): sp.set_color(self._CS)

        # Gráfica 1: Velocidad (fila 0, columnas 0-1)
        self.ax1 = self.fig.add_subplot(gs[0, :2])
        _sty(self.ax1, "Velocidad vs Tiempo", "v [m/s]")
        self.line_sp,   = self.ax1.plot([], [], "--", color="#f0883e", lw=1.5,
                                         label=f"Setpoint")
        self.line_pid,  = self.ax1.plot([], [], "-",  color="#3fb950", lw=2.2,
                                         label="Con PID")
        self.line_open, = self.ax1.plot([], [], "-",  color="#f85149", lw=1.8,
                                         label="Sin ctrl", alpha=0.7)
        self.ax1.legend(facecolor="#21262d", edgecolor=self._CS,
                        labelcolor="#e6edf3", fontsize=8, loc="upper left")

        # Gráfica 2: Error (fila 0, columna 2)
        self.ax2 = self.fig.add_subplot(gs[0, 2])
        _sty(self.ax2, "Error e(k)", "Error [m/s]")
        self.ax2.axhline(0, color=self._CT, lw=0.7, ls=":")
        self.line_err, = self.ax2.plot([], [], "-", color="#d2a8ff", lw=1.8)

        # Gráfica 3: Integral [M1] (fila 1, columna 0)
        self.ax3 = self.fig.add_subplot(gs[1, 0])
        _sty(self.ax3, "Acumulador Integral [M1]", "∑e·dt")
        self.ax3.axhline(0, color=self._CT, lw=0.6, ls=":")
        i_max = model.u_max / model.Ki if model.Ki != 0 else 1e9
        self.ax3.axhline( i_max, color="#f85149", lw=0.9, ls="--",
                          alpha=0.6, label=f"±{i_max:.1f}")
        self.ax3.axhline(-i_max, color="#f85149", lw=0.9, ls="--", alpha=0.6)
        self.line_int, = self.ax3.plot([], [], "-", color="#ffa657", lw=1.8)
        self.ax3.legend(facecolor="#21262d", edgecolor=self._CS,
                        labelcolor="#e6edf3", fontsize=7)

        # Gráfica 4: Derivada filtrada [M3] (fila 1, columna 1)
        self.ax4 = self.fig.add_subplot(gs[1, 1])
        _sty(self.ax4, f"Derivada Filtrada EMA α={model.alpha} [M3]", "d(k)")
        self.ax4.axhline(0, color=self._CT, lw=0.6, ls=":")
        self.line_df, = self.ax4.plot([], [], "-", color="#58a6ff", lw=1.8)

        # Panel estado (fila 1, columna 2)
        self.ax5 = self.fig.add_subplot(gs[1, 2])
        self.ax5.set_facecolor(self._CB)
        self.ax5.axis("off")
        self._info = self.ax5.text(
            0.05, 0.97, "", transform=self.ax5.transAxes,
            fontsize=8.5, va="top", color="#e6edf3", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="#21262d", edgecolor=self._CS, alpha=0.9)
        )

        # ── Leyenda de controles ──────────────────────────────────────────
        self._ctrl_legend = self.ax5.text(
            0.05, 0.28, self._controls_text(), transform=self.ax5.transAxes,
            fontsize=7.5, va="top", color="#8b949e", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#161b22", edgecolor=self._CS, alpha=0.85)
        )

        # ── Botones ───────────────────────────────────────────────────────
        # Posiciones: [left, bottom, width, height] en coordenadas de figura
        btn_y  = 0.035
        btn_h  = 0.055
        btn_w  = 0.13
        gap    = 0.155
        starts = [0.06, 0.06 + gap, 0.06 + 2*gap, 0.06 + 3*gap, 0.06 + 4*gap]

        btn_style = dict(color="#21262d", hovercolor="#30363d")

        ax_b0 = self.fig.add_axes([starts[0], btn_y, btn_w, btn_h])
        ax_b1 = self.fig.add_axes([starts[1], btn_y, btn_w, btn_h])
        ax_b2 = self.fig.add_axes([starts[2], btn_y, btn_w, btn_h])
        ax_b3 = self.fig.add_axes([starts[3], btn_y, btn_w, btn_h])
        ax_b4 = self.fig.add_axes([starts[4], btn_y, btn_w, btn_h])

        self.btn_sp_up   = Button(ax_b0, "▲ Subir Referencia", **btn_style)
        self.btn_sp_down = Button(ax_b1, "▼ Bajar Referencia", **btn_style)
        self.btn_kick_p  = Button(ax_b2, "⚡ Acelerar +8 m/s", **btn_style)
        self.btn_kick_n  = Button(ax_b3, "💥 Frenar  −8 m/s",  **btn_style)
        self.btn_reset   = Button(ax_b4, "↺  Reiniciar",        **btn_style)

        # Colorear texto de botones
        for btn, col in [
            (self.btn_sp_up,   "#f0883e"),
            (self.btn_sp_down, "#f0883e"),
            (self.btn_kick_p,  "#3fb950"),
            (self.btn_kick_n,  "#f85149"),
            (self.btn_reset,   "#d2a8ff"),
        ]:
            btn.label.set_color(col)
            btn.label.set_fontsize(9)

        # ── Conectar botones ──────────────────────────────────────────────
        self.btn_sp_up.on_clicked(  lambda _: self._queue("sp_up"))
        self.btn_sp_down.on_clicked(lambda _: self._queue("sp_down"))
        self.btn_kick_p.on_clicked( lambda _: self._queue("kick_p"))
        self.btn_kick_n.on_clicked( lambda _: self._queue("kick_n"))
        self.btn_reset.on_clicked(  lambda _: self._queue("reset"))

        # ── Conectar teclado ──────────────────────────────────────────────
        # mpl_connect registra on_key para todos los eventos 'key_press_event'
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # ── Detectar cierre de ventana → detener simulación ───────────────
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        plt.ion()
        plt.show()

    # ── Callbacks de interacción ──────────────────────────────────────────────
    def _queue(self, action: str):
        """Encola una acción para que el bucle principal la ejecute en el siguiente tick."""
        self.pending_action = action

    def _on_key(self, event):
        """
        Callback de teclado (mpl_connect 'key_press_event').
        Mapeo de teclas:
            up    → sube setpoint
            down  → baja setpoint
            a     → kick positivo
            z     → kick negativo
            r     → reset
            q/escape → cerrar
        """
        mapping = {
            "up"    : "sp_up",
            "down"  : "sp_down",
            "a"     : "kick_p",
            "z"     : "kick_n",
            "r"     : "reset",
            "q"     : "quit",
            "escape": "quit",
        }
        key = (event.key or "").lower()
        if key in mapping:
            self._queue(mapping[key])

    def _on_close(self, event):
        """Se llama cuando el usuario cierra la ventana de matplotlib."""
        self.running = False

    def consume_action(self):
        """
        El bucle principal llama a esto en cada tick.
        Si hay una acción pendiente (teclado o botón), la ejecuta sobre el modelo.
        Retorna True si se debe detener la simulación.
        """
        action = self.pending_action
        self.pending_action = None

        if action is None:
            return False
        if action == "quit":
            self.running = False
            return True
        if action == "sp_up":
            self.model.apply_setpoint_up()
        elif action == "sp_down":
            self.model.apply_setpoint_down()
        elif action == "kick_p":
            self.model.apply_kick_positive()
        elif action == "kick_n":
            self.model.apply_kick_negative()
        elif action == "reset":
            self.model.apply_reset()
        return False

    # ── Actualización de gráficas ─────────────────────────────────────────────
    def update(self):
        """
        Refresca todas las gráficas con los datos más recientes.
        Usa ventana deslizante de WINDOW_SIZE pasos para no acumular
        millones de puntos en simulaciones largas.
        """
        model = self.model
        t   = model.time_data
        if len(t) == 0:
            return

        # ── Ventana deslizante ─────────────────────────────────────────────
        # Mostramos solo los últimos WINDOW_SIZE puntos en pantalla.
        # El historial completo sigue en el DataCollector para análisis.
        n   = len(t)
        sl  = slice(max(0, n - WINDOW_SIZE), n)   # índice de ventana

        tw   = t[sl]
        vp   = model.vel_pid_data[sl]
        vo   = model.vel_open_data[sl]
        err  = model.error_pid_data[sl]
        itg  = model.integral_data[sl]
        df_  = model.d_filtered_data[sl]

        # Setpoint (puede haber cambiado)
        sp = model.setpoint

        self.line_sp.set_data([tw[0], tw[-1]], [sp, sp])
        self.line_pid.set_data(tw, vp)
        self.line_open.set_data(tw, vo)
        self.line_err.set_data(tw, err)
        self.line_int.set_data(tw, itg)
        self.line_df.set_data(tw, df_)

        for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
            ax.relim()
            ax.autoscale_view()

        # ── Dibujar líneas de perturbación (solo las nuevas) ──────────────
        self._draw_disturbances(tw)

        # ── Panel de estado ───────────────────────────────────────────────
        pid_agent = model.schedule.agents[0]
        self._info.set_text(
            f"  t = {model.current_time:.1f} s\n"
            f"  Setpoint  : {sp:.1f} m/s\n"
            f"  Vel PID   : {pid_agent.velocity:.3f} m/s\n"
            f"  Error     : {err[-1]:.4f} m/s\n"
            f"  Accel     : {pid_agent.acceleration:.3f} m/s²\n"
            f"  Integral  : {itg[-1]:.3f}\n"
            f"  D_filtrada: {df_[-1]:.4f}\n\n"
            f"  Kp={model.Kp}  Ki={model.Ki}  Kd={model.Kd}\n"
            f"  α={model.alpha}  u_max={model.u_max}\n"
            f"  σ={model.noise_std} m/s\n"
            f"  Perturbaciones: {len(model.disturbances)}"
        )

        self.fig.canvas.draw_idle()
        plt.pause(0.001)    # cede control al event-loop → refresca ventana

    def _draw_disturbances(self, tw: np.ndarray):
        """
        Dibuja una línea vertical anotada en cada gráfica para cada perturbación
        que aún no haya sido dibujada y que esté dentro de la ventana visible.
        """
        t_min = tw[0] if len(tw) else 0.0
        t_max = tw[-1] if len(tw) else 0.0

        for step_k, label, color in self.model.disturbances:
            if step_k in self._drawn_disturbances:
                continue
            t_event = step_k * self.model.dt
            if not (t_min <= t_event <= t_max):
                continue   # todavía fuera de la ventana visible

            for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
                ax.axvline(t_event, color=color, lw=1.2, ls="--", alpha=0.75)
                ax.text(t_event, ax.get_ylim()[1], f" {label}",
                        color=color, fontsize=7, va="top",
                        rotation=90, clip_on=True)
            self._drawn_disturbances.add(step_k)

    @staticmethod
    def _controls_text() -> str:
        return (
            "  ── Controles ────────────────\n"
            "  [↑] / ▲  Subir referencia\n"
            "  [↓] / ▼  Bajar referencia\n"
            "  [A] / ⚡  Acelerar +8 m/s\n"
            "  [Z] / 💥  Frenar  −8 m/s\n"
            "  [R] / ↺   Reiniciar\n"
            "  [Q/Esc]   Salir"
        )


# ════════════════════════════════════════════════════════════════════════════
#  4. BUCLE PRINCIPAL — SIMULACIÓN INDEFINIDA [v3 - FIX 1]
# ════════════════════════════════════════════════════════════════════════════
def run_simulation():
    """
    Bucle infinito: avanza la simulación de MESA indefinidamente
    hasta que el usuario cierre la ventana o pulse Q/Esc.

    Cómo funciona el bucle infinito:
    ──────────────────────────────────
    En vez de `for step in range(N)`, usamos `while plotter.running`.
    La condición `running` se pone en False desde:
        · _on_close()  → usuario cierra la ventana de matplotlib
        · _on_key()    → usuario pulsa Q o Escape
    En cada iteración:
        1. model.step()             → avanza la física + DataCollector
        2. plotter.consume_action() → aplica perturbación pendiente (si hay)
        3. plotter.update()         → refresca gráficas y cede evento-loop
    """

    # ── Parámetros ────────────────────────────────────────────────────────
    SETPOINT   = 20.0
    DT         = 0.05
    KP, KI, KD = 5.0, 1.2, 0.8
    DAMPING    = 0.3
    U_MAX      = 50.0
    ALPHA      = 0.3
    NOISE_STD  = 0.4

    print("=" * 62)
    print("  MESA PID Simulation v3.0 — Simulación Indefinida")
    print("=" * 62)
    print(f"  Setpoint inicial : {SETPOINT} m/s")
    print(f"  Kp={KP}  Ki={KI}  Kd={KD}")
    print(f"  Ventana visible  : {WINDOW_SIZE * DT:.1f} s  ({WINDOW_SIZE} pasos)")
    print()
    print("  CONTROLES:")
    print("    Teclado [↑]/[↓] o botones ▲▼  → subir/bajar referencia ±5 m/s")
    print("    Teclado [A]      o botón  ⚡   → acelerar +8 m/s (empuje)")
    print("    Teclado [Z]      o botón  💥   → frenar −8 m/s (frenazo)")
    print("    Teclado [R]      o botón  ↺    → reiniciar")
    print("    Teclado [Q/Esc] o cerrar ventana → terminar")
    print("=" * 62)

    model   = VehicleModel(
        Kp=KP, Ki=KI, Kd=KD, dt=DT,
        setpoint=SETPOINT, damping=DAMPING,
        u_max=U_MAX, alpha=ALPHA, noise_std=NOISE_STD
    )
    plotter = RealTimePlotter(model=model)

    # ── [v3 - FIX 1] Bucle indefinido ────────────────────────────────────
    while plotter.running:
        model.step()                    # física + DataCollector

        should_stop = plotter.consume_action()   # perturbación pendiente
        if should_stop:
            break

        plotter.update()                # refresca gráficas

        # Log de consola cada 100 pasos (~5 s de simulación)
        if model.step_count % 100 == 0:
            print(f"  t={model.current_time:7.1f}s | "
                  f"SP={model.setpoint:.1f} | "
                  f"v_PID={model.vel_pid_data[-1]:.3f} | "
                  f"err={model.error_pid_data[-1]:.4f} | "
                  f"∫={model.integral_data[-1]:.3f}")

    print(f"\n  Simulación terminada en t={model.current_time:.1f}s  "
          f"({model.step_count} pasos).")
    print(f"  Perturbaciones aplicadas: {len(model.disturbances)}")
    if model.disturbances:
        for step_k, label, _ in model.disturbances:
            print(f"    t={step_k*DT:.2f}s  →  {label}")

    plt.ioff()
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_simulation()