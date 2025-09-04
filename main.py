import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# -------------------------------
# Funkcje symulacji
# -------------------------------
def f(t, x, u, A, B):
    return A @ x + B @ np.atleast_1d(u)

def output(x, u, C, D):
    return C @ x + D @ np.atleast_1d(u)

def euler_step(x, u, t, h, A, B):
    return x + h * f(t, x, u, A, B)

def rk4_step(x, u, t, h, A, B):
    k1 = f(t, x, u, A, B)
    k2 = f(t + h/2, x + h/2 * k1, u, A, B)
    k3 = f(t + h/2, x + h/2 * k2, u, A, B)
    k4 = f(t + h,   x + h * k3,   u, A, B)
    return x + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

# -------------------------------
# Funkcje sygnałów wejściowych
# -------------------------------
def u_square(t, amplitude, frequency, phase, duty):
    cycle_pos = ((t + phase) / (1/frequency)) % 1
    return amplitude if cycle_pos < duty else -amplitude

def u_sawtooth(t, amplitude, frequency, phase):
    return amplitude * (2 * ((t + phase) / (1/frequency) % 1) - 1)

def u_harmonic(t, amplitude, frequency, phase):
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)

# -------------------------------
# GUI
# -------------------------------
class SimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Symulator układu mechanicznego")

        # Panel kontrolny
        control_frame = ttk.Frame(root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Parametry układu
        # Początkowe wartości parametrów
        ttk.Label(control_frame, text="Parametry układu").pack()
        self.J1 = tk.DoubleVar(value=10.0)
        self.J2 = tk.DoubleVar(value=10.0)
        self.b1 = tk.DoubleVar(value=10.0)
        self.b2 = tk.DoubleVar(value=10.0)
        self.n1 = tk.DoubleVar(value=10.0)
        self.n2 = tk.DoubleVar(value=20.0)

        for text, var, unit in [("J\u2081", self.J1, "kg·m²"),
                        ("J\u2082", self.J2, "kg·m²"),
                        ("b\u2081", self.b1, "N·m·s/rad"),
                        ("b\u2082", self.b2, "N·m·s/rad"),
                        ("n\u2081", self.n1, "-"),
                        ("n\u2082", self.n2, "-")]:
            frame = ttk.Frame(control_frame)
            frame.pack(anchor="w")
            ttk.Label(frame, text=f"{text}:").pack(side=tk.LEFT)
            ttk.Entry(frame, textvariable=var, width=8).pack(side=tk.LEFT)
            ttk.Label(frame, text=unit).pack(side=tk.LEFT)


        ttk.Label(control_frame, text="").pack()  # separator

        # Parametry sygnału wejściowego
        ttk.Label(control_frame, text="Parametry sygnału wejściowego").pack()
        self.signal_type = tk.StringVar(value="sinus")
        ttk.Radiobutton(control_frame, text="Harmoniczny", variable=self.signal_type, value="sinus").pack(anchor="w")
        ttk.Radiobutton(control_frame, text="Prostokątny", variable=self.signal_type, value="square").pack(anchor="w")
        ttk.Radiobutton(control_frame, text="Trójkątny", variable=self.signal_type, value="sawtooth").pack(anchor="w")

        self.amplitude = tk.DoubleVar(value=1.0)
        self.frequency = tk.DoubleVar(value=0.5)
        self.phase = tk.DoubleVar(value=0.0)
        self.duty = tk.DoubleVar(value=0.5)

        for text, var, unit in [("Amplituda", self.amplitude, "N·m"),
                        ("Częstotliwość", self.frequency, "Hz"),
                        ("Faza", self.phase, "rad"),
                        ("Wypełnienie", self.duty, "-")]:
            frame = ttk.Frame(control_frame)
            frame.pack(anchor="w")
            ttk.Label(frame, text=f"{text}:").pack(side=tk.LEFT)
            ttk.Entry(frame, textvariable=var, width=8).pack(side=tk.LEFT)
            ttk.Label(frame, text=unit).pack(side=tk.LEFT)

        ttk.Label(control_frame, text="").pack()  # separator

        # Parametry symulacji
        ttk.Label(control_frame, text="Parametry symulacji").pack()
        self.t0 = tk.DoubleVar(value=0.0)
        self.tf = tk.DoubleVar(value=10.0)
        self.h = tk.DoubleVar(value=0.01)

        # Warunki początkowe
        self.x10 = tk.DoubleVar(value=0.0)  # x1(0)
        self.x20 = tk.DoubleVar(value=0.0)  # x2(0)

        for text, var, unit in [("Początek symulacji", self.t0, "s"),
                        ("Koniec symulacji", self.tf, "s"),
                        ("Skok", self.h, "-"),
                        ("\u03B8\u2081(0)", self.x10, "°"),
                        ("\u03C9\u2081(0)", self.x20, "rad/s")]:
            frame = ttk.Frame(control_frame)
            frame.pack(anchor="w")
            ttk.Label(frame, text=f"{text}:").pack(side=tk.LEFT)
            ttk.Entry(frame, textvariable=var, width=8).pack(side=tk.LEFT)
            ttk.Label(frame, text=unit).pack(side=tk.LEFT)

        ttk.Button(control_frame, text="Symuluj", command=self.run_simulation).pack(pady=10)

        # Miejsce na wykresy
        plot_frame = ttk.Frame(root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figure, self.axs = plt.subplots(3, 1, figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Miejsce na obraz
        image_frame = ttk.Frame(control_frame)
        image_frame.pack(pady=10)

        # Wczytaj obraz układu
        self.image = Image.open("uklad.png")
        self.image = self.image.resize((300, 250), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.image)

        self.image_label = ttk.Label(image_frame, image=self.photo)
        self.image_label.pack()

        # Dodanie toolbara do powiększania/zmniejszania wykresów
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run_simulation(self):
        # Wczytanie parametrów
        J1, J2, b1, b2, n1, n2 = self.J1.get(), self.J2.get(), self.b1.get(), self.b2.get(), self.n1.get(), self.n2.get()
        Jeq = J1 + J2 * (n1/n2)**2
        beq = b1 + b2 * (n1/n2)**2

        # Model stanowy
        A = np.array([[0, 1], [0, -beq/Jeq]])
        B = np.array([[0], [1/Jeq]])
        C = np.array([[(n1/n2), 0], [0, (n1/n2)]])
        D = np.array([[0], [0]])
        
        # Inicjalizacja symulacji
        t0, tf, h = self.t0.get(), self.tf.get(), self.h.get()
        t_vals = np.arange(t0, tf, h)

        x0 = np.array([np.radians(self.x10.get()), self.x20.get()])
        x_euler = [x0]
        x_rk4 = [x0]

        # Wybór sygnału wejściowego
        if self.signal_type.get() == "square":
            u_func = lambda t: u_square(t, self.amplitude.get(), self.frequency.get(), self.phase.get(), self.duty.get())
        elif self.signal_type.get() == "sawtooth":
            u_func = lambda t: u_sawtooth(t, self.amplitude.get(), self.frequency.get(), self.phase.get())
        else:
            u_func = lambda t: u_harmonic(t, self.amplitude.get(), self.frequency.get(), self.phase.get())

        #Inicjalizacja wyjścia
        u_vals = []
        y_euler = [output(x0, u_func(t0), C, D)]
        y_rk4 = [output(x0, u_func(t0), C, D)]

        for t in t_vals[:-1]:
            u = u_func(t)
            u_vals.append(u)
            x_euler.append(euler_step(x_euler[-1], u, t, h, A, B))
            x_rk4.append(rk4_step(x_rk4[-1], u, t, h, A, B))
            y_euler.append(output(x_euler[-1], u, C, D))
            y_rk4.append(output(x_rk4[-1], u, C, D))

        x_euler = np.array(x_euler)
        x_rk4 = np.array(x_rk4)
        y_euler = np.array(y_euler)
        y_rk4 = np.array(y_rk4)

        # Update wykresów
        for ax in self.axs:
            ax.clear()

        # Wykres sygnału wejściowego
        self.axs[0].plot(t_vals, [u_func(t) for t in t_vals], label="u(t)")
        self.axs[0].set_title("Sygnał wejściowy u(t) = T\u2098 [N·m]")
        self.axs[0].set_ylabel("T\u2098(t) [N·m]")
        self.axs[0].legend()
        self.axs[0].grid()
        # Wykres wyjścia theta
        self.axs[1].plot(t_vals, np.degrees(y_euler[:,0]), label="Euler - \u03B8\u2082 [°]")
        self.axs[1].plot(t_vals, np.degrees(y_rk4[:,0]), "--", label="RK4 - \u03B8\u2082 [°]")
        self.axs[1].set_title("Wyjście \u03B8\u2082 [°]")
        self.axs[1].set_ylabel("Kąt [°]")
        self.axs[1].legend()
        self.axs[1].grid()
        # Wykres wyjścia omega
        self.axs[2].plot(t_vals, y_euler[:,1], label="Euler - \u03C9\u2082 [rad/s]")
        self.axs[2].plot(t_vals, y_rk4[:,1], "--", label="RK4 - \u03C9\u2082 [rad/s]")
        self.axs[2].set_title("Wyjście \u03C9\u2082 [rad/s]")
        self.axs[2].set_ylabel("Prędkość [rad/s]")
        self.axs[2].legend()
        self.axs[2].grid()

        self.figure.tight_layout()
        self.canvas.draw()

# -------------------------------
# Działanie aplikacji
# -------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SimulatorApp(root)

    # Zamykanie aplikacji
    def on_closing():
        plt.close('all')
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()