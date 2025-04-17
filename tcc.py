""" Circuit Shaper Simulator """
import os
import numpy as np
import random
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sympy as sp
from sympy import Eq
from sympy import fraction
from sympy.abc import s
from scipy import signal
import scipy.signal as signal
from scipy.stats import pearsonr

# Constants
xlabels = ["C0", "Ca", "Cb", "Cc", "La", "Lb", "Lc", "RL"]
sigma = 3

## Functions ##
def transpose(matrix):
    """
    Transpose a given matrix, swapping rows with columns.

    Parameters:
    - matrix (list of lists): A rectangular 2D list where each sublist represents a row.

    Returns:
    - list of lists: A new matrix with rows and columns transposed.

    ################################
    # all_x_coord transpose of MC
    # all_y_real transpose of real_pt
    # all_y_imag transpose of imag_pt
    ################################
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

def save_figure(file_name, directory='figs', ext='png', dpi=300, add_date=True):
    """
    Saves the current matplotlib figure to a specified folder.

    Parameters:
    - filename (str): base name of the file (without extension)
    - folder (str): directory where the file will be saved
    - extension (str): file extension ('png', 'jpg', 'pdf' etc.)
    - dpi (int): image resolution
    - add_date (bool): if True, adds timestamp to the file name
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    timestamp = datetime.now().strftime('%d-%m-%Y') if add_date else ''
    full_name = f"{file_name}_{timestamp}.{ext}" if add_date else f"{file_name}.{ext}"
    full_path = os.path.join(directory, full_name)

    plt.savefig(full_path, format=ext, dpi=dpi, bbox_inches='tight')
    print(f"Saved in: {full_path}")

def scatter(x_coords, y, pole_name, pol='\n', size=10, save_fig=False):

    _, axs = plt.subplots(2, 4, figsize=(size,size))

    # Main title with the name of the pole
    plt.suptitle(f"Polo {pole_name}{pol}", fontweight="bold")

    for i, (xi, label) in enumerate(zip(x_coords[:8], xlabels)):
        row, col = divmod(i, 4) # return tupla (a // b, a % b)
        axs[row, col].scatter(xi, y)
        axs[row, col].set_title(f"Scatter Plot {i+1}")
        axs[row, col].set_xlabel(label, fontweight="bold")
        axs[row, col].set_ylabel("Variação do Polo")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_fig:
        file_name = f"Scatter_polo_{pole_name}".replace(" ", "_")
        save_figure(file_name, directory='Scatter')

def correlacao(x, y, pole_name, pol='\n', size=10, save_fig=False):
    
    corr_coefs = [pearsonr(xi, y)[0] for xi in x]

    x_corr = {key: abs(coef) for key, coef in zip(xlabels, corr_coefs) }
    
    sorted_coefficients = sorted(corr_coefs, key=abs)
    x_coords_labels = sorted(xlabels, key=x_corr.__getitem__)

    # Plot the horizontal bar graph of Pearson's correlation coefficients
    plt.figure(figsize=(size, size))
    plt.xlabel("Influência do Parâmetro")
    plt.title(f"Polo {pole_name}{pol}", fontweight="bold")
    plt.grid(axis="x")
    plt.barh(x_coords_labels, sorted_coefficients, color="darkblue")

    bars = plt.barh(x_coords_labels, sorted_coefficients, color="gray")

    # Add the values ​​in the bars
    for bar, value in zip(bars, sorted_coefficients):
        plt.text(
            bar.get_width()/2, # Horizontally centered position of the bar
            bar.get_y() + bar.get_height() / 2, # Vertically centered on the bar
            f"{value:.2f}", # Formatted to 2 decimal places
            va="center", # Centered vertical alignment
            ha="center", # Left horizontal alignment
            fontsize=11,  
            color="black" , 
            fontweight='bold' 
        )

    if save_fig:
        file_name = f"Correlacao_polo_{pole_name}".replace(" ", "_")
        save_figure(file_name, directory='Correlacao')

def pole_map(all_func_poles, save_fig=False):

    # Remove the two largest poles from each row (poles generated by tau1 and tau2)
    filtered_poles = [linha[2:] for linha in np.sort(all_func_poles, axis=1)]  
    filtered_poles = np.array(filtered_poles)

    _, ax = plt.subplots(figsize=(7, 4))

    # Separation of real and imaginary parts
    real_pt = np.real(filtered_poles)  
    imag_pt = np.imag(filtered_poles)  

    # Ellipses auxiliary variables
    mx = np.mean(real_pt, axis=0) # mean real
    lx = np.std(real_pt, axis=0) # standard deviation real
    my = np.mean(imag_pt, axis=0) # mean imaginaria
    ly = np.std(imag_pt, axis=0) # imaginary standard deviation

    # Standard deviation ellipses
    for i in range(len(real_pt[0])):
        elipse = Ellipse(
            xy=(mx[i], my[i]),
            width=8.5 * lx[i],
            height=15.5 * ly[i],
            angle=0,
            alpha=0.5,
            facecolor="grey"
        )
        ax.add_patch(elipse)

    # Plot of poles and zero
    ax.scatter(real_pt, imag_pt, marker=".", label="Polos") 
    ax.scatter(real_pt[0], imag_pt[0], marker="x", label="Polos Nominais")
    ax.scatter(0, 0, s=13, facecolor='none', edgecolor='red', linewidth=1, label="Z1")

    # Nomeando os polos: p1 a p6
    desloc = 0
    width_x = plt.xlim()[1] - plt.xlim()[0]
    for i in range(6):
        ax.text(real_pt[0][i] + desloc * width_x, imag_pt[0][i] + desloc, f'p{i+1}', fontsize=9, color='blue')

    # Axis e grid
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True)

    # Titles and subtitles
    ax.set_title("Plano Complexo")
    ax.set_xlabel("Parte Real")
    ax.set_ylabel("Parte Imaginária")
    ax.legend()
    plt.tight_layout()

    if save_fig:
        save_figure('Mapa_polos')

    plt.show(block=False)  # does not block execution
    plt.pause(0.5)
    return transpose(real_pt), transpose(imag_pt)

def config_plot_pulso(titles, labelx, ylabel, xlim, ylim=None):
    """
    Configures the visual appearance of a Matplotlib plot for pulse signals.

    Adds title, axis labels, limits, axis lines at zero, and a grid to the plot.

    Args:
        titulo (str): The title of the plot.
        labelx (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        xlim (tuple): Tuple containing the minimum and maximum values for the x-axis (x_min, x_max).
        ylim (tuple, optional): Tuple containing the minimum and maximum values for the y-axis (y_min, y_max). Defaults to None.

    Returns:
        None
    """
    plt.title(titles)
    plt.xlabel(labelx)
    plt.ylabel(ylabel)
    plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.grid(True)
    plt.tight_layout()

def plot_pulso(t, y, sigma, save_fig=False):
    
    # Mean and standard deviation calculation
    y = np.array(y)
    ymed = np.mean(y, axis=0)
    desv_pad = np.std(y, axis=0)

    # --- 1. Pulso with variations ---
    [plt.plot(t, yi, color="b", linewidth=2) for yi in y]
    config_plot_pulso("Pulso com Variação", "Tempo", "Intensidade do Pulso", (-0.25e-6, 0.5e-6))
    plt.show(block=False)
    plt.pause(0.5)
    plt.clf()

    # --- 2. Pulso without variations ---
    plt.plot(t, y[0], color="b", linewidth=2)
    config_plot_pulso("Pulso Sem Variação", "Tempo", "Intensidade do Pulso", (-0.25e-6, 0.5e-6))
    plt.show(block=False)
    plt.pause(0.5)
    plt.clf()

    # --- 3. Uncertainty Band ---
    banda_sup = ymed + desv_pad * sigma
    banda_inf = ymed - desv_pad * sigma

    plt.plot(y[0], label="Pulso com Valores Nominais", color="b", linewidth=2)
    plt.plot(ymed, label="Média dos Erros", linestyle="--", color="orange", linewidth=2)
    plt.plot(banda_sup, label="Banda Superior", linestyle="-.", color="g", linewidth=2)
    plt.plot(banda_inf, label="Banda Inferior", linestyle="-.", color="r", linewidth=2)
    plt.fill_between(range(len(banda_sup)), banda_sup, banda_inf, color="gray", alpha=0.5)

    config_plot_pulso("Bandas de Incerteza", "Tempo", "Intensidade Pulso", (-0.1, 10), (-0.2, 1.25))
    plt.legend()

    if save_fig:
        save_figure("Banda_incerteza")

    plt.show(block=False)  
    plt.pause(0.5)
    plt.clf()

def MonteCarlo_iteration(iterations, erro, nominal_values, FT, t):
    """
    iter = 0 represents pure/error-free signal value
    MonteCarlo[0] = actual/error-free signal value

    FT = Transfer Function
    t1 = time vector
    iterations = number of iterations
    erro = percentage error of circuit components
    nominal_values = nominal values of circuit components
    """

    # Store all the poles of the iterations
    all_poles = [] 

    # Store the values ​​with error of each iteration
    MonteCarlo = [] 

    # Store all the FPs added of each iteration # list of all the graphs added with errors
    y_out = [] 
    
    # Helper to store the values ​​of y
    y1 = [] 


    for iter in range(iterations):

        if iterations < 50:
            print(f'{iter+1}/{iterations}')
        elif iter % 100 == 0:
            print(f'iter: {iter}')

        # Random values; # error range; maximum error from -e% to +e%
        xa = [] # FP function of the summed iteration
        Cval = [] # List of components with changed values

        # Changing element values ​​without changing tau1 and tau2
        Cval = [value * (random.gauss(0, erro[idx])) + value if iter != 0 and idx < len(nominal_values) - 2 else value for idx, value in enumerate(nominal_values) ]

        # Saving component variations
        MonteCarlo.append(Cval[:-2])  

        H = FT
        for variavel, v in zip(Cord, Cval):
            H = H.subs(variavel, v)

        # Separating numerator from denominator
        N_H, D_H = fraction(H)

        """RESIDUOS E POLOS"""

        coefs_num = []  # reset variable
        coefs_den = []  # reset variable

        coefs_num = sp.Poly(N_H, s).all_coeffs()  # get coefs
        coefs_den = sp.Poly(D_H, s).all_coeffs()  # get coefs

        # frações parciais
        residuos, polos, b0 = [], [], []
        residuos, polos, b0 = signal.residue(coefs_num, coefs_den)

        # Saving all poles
        all_poles.append(polos)

        # Correção do residuos (tirando a parte img dos residuos reais)
        # for polo1, residuo1 in zip(polos, residuos):
        #     if polo1.imag == 0:
        #         residuo1 = residuo1.real

        """LAPLACE INVERSA E GRAFICOS"""
       
        for enum, (polo, residuo) in enumerate(zip(polos, residuos)):

            # Removing the img part (0j) of the real residues
            # if polo.imag == 0:
            #     residuo = residuo.real
            #     polo = polo.real

            # Check if the imaginary part is zero
            if polo.imag == 0:
                "EXPONENCIAIS"

                # Removing the img part (0j) of the real residues
                A = residuo.real
                d =  polo.real
                x = A * np.exp(d * t)

                xa.append(x)

            else:
                "SENOS E COSSENOS"

                pol_1 = polos[enum - 1]  # conjugate

                if polo != pol_1 and polo != np.conjugate(pol_1):

                    a1 = polo.real  
                    b1 = abs(polo.imag)  

                    Mod = abs(residuos[enum])
                    fase = np.angle(residuos[enum])  # fase in rad

                    # FP term
                    x = 2 * Mod * np.exp(a1 * t) * np.cos(b1 * t + fase)

                    xa.append(x)

        "SUMMING THE TERMS"

        if iter != 0:  
            y1 = sum(xa).real / maxs
            y_out.append(y1)
            plt.xlim(-0.1e-6, 0.5e-6)
            plt.ylim(-0.25, 1.2)
            plt.axhline(0, color="black", linewidth=0.65)
            plt.axvline(0, color="black", linewidth=0.65)
            plt.grid(True)
            plt.plot(t, y1, color="blue")
        else:
            sinal0 = sum(xa).real
            maxs = max(abs(sinal0))
            y1 = sinal0 / maxs  # largest module/ normalize
            y_out.append(y1)
            plt.plot(t, y1, color="black")

    return transpose(MonteCarlo), all_poles, y_out

def Pearson8(all_y_coord, x_coords, pole_name, stop=0):
    
    for enum, y_coord in enumerate(all_y_coord):
        try:
            if stop==enum and stop!=0:
                break

            scatter(x_coords, y_coord, pole_name=pole_name[enum])
            correlacao(x_coords, y_coord, pole_name=pole_name[enum])

        except Exception:
            print(f"Erro ao gerar gráfico para {pole_name[enum]}")
            continue

    plt.show(block=False)  
    plt.pause(0.5) 

def histogram(data, pole_name, save_fig=False):
    """
    Histogram of the poles (with the values of y_real and y_imag)
    """

    for enum, y_coord in enumerate(data):
        mean = np.mean(y_coord)
        deviation = np.std(y_coord)
        plt.hist(y_coord, bins=10, color='darkblue', edgecolor='black', alpha=0.7)
        plt.axvline(mean + deviation, color='red', linestyle='dashed', linewidth=2, label='Standard Deviation')
        plt.axvline(mean - deviation, color='red', linestyle='dashed', linewidth=2, label='Standard Deviation')

        # Text with σ
        plt.text(mean + deviation + 0.005, max(np.histogram(y_coord, bins=10)[0]) * 0.85,
        fr'$\sigma$ = {sigma}%', color='red', fontsize=12)

        # Titles and labels
        plt.title(f"Distribution of {pole_name[enum]} Values")
        # plt.legend()
        plt.xlabel("Distribution of Pole Values")
        plt.ylabel("Count")

        # Pretty layout
        plt.grid(True)
        plt.tight_layout()

        if save_fig:
            file_name = f"Hist_polo_{pole_name}".replace(" ", "_")
            save_figure(file_name, directory='Histogramas')
        plt.show(block=False)
    plt.pause(0.5)

## Definition of constants ##
tau_1, tau_2, Vo, Vi = sp.symbols("tau_1 tau_2 Vo Vi")
CC0, C1, C2, C3, C4, C5 = sp.symbols("CC0 C1 C2 C3 C4 C5")
R3, R1, R2, RL = sp.symbols("R3 R1 R2 RL")
L1, L2, L3, L4, L5, L6 = sp.symbols("L1 L2 L3 L4 L5 L6")
I1, I2, I3, I4, I5, I6 = sp.symbols("I1 I2 I3 I4 I5 I6")

# Poles names
pole_names_real = ['P1 Real','P2 Real','P3 Real','P4 Real','P5 Real','P6 Real']
pole_names_imag = ['P1 Imag','P2 Imag','P3 Imag','P4 Imag','P5 Imag','P6 Imag']

# ckt components in order (old values)
Cord = [CC0, C1, C2, C3, L1, L2, L3, RL, tau_1, tau_2]

# Nominal values of Cord elements in order
Cval = [
    100e-9,  # CC0
    120e-12,  # Ca = C1 + C2
    130e-12,  # Cb = C3 + C4
    83e-12,  # Cc = C5 + C6
    2.48e-6,  # La = L1 + L2
    1.6e-6,  # Lb = L3 + L4
    0.78e-6,  # Lc = L5 + L6
    138.8338,  # RL = (R1 + R2) //  R3
    3.1046e-09,  # tau_2
    6.5798e-09,  # tau_1
]

# Plot limit with 400 points with distance of 25*10^-9 between them
t1 = np.arange(0, 400) * 25 * 10**-9

# ckt equations
eqn1 = Eq(I1 / (CC0 * s) + (I1 - I2) / (C1 * s), Vi)
eqn2 = Eq((I2 - I3) / (C2 * s) - (I1 - I2) / (C1 * s) + L1 * s * I2, 0)
eqn3 = Eq((I3 - I4) / (C3 * s) - (I2 - I3) / (C2 * s) + L2 * s * I3, 0)
eqn4 = Eq(I4 * RL - (I3 - I4) / (C3 * s) + L3 * s * I4, 0)
eqn5 = Eq(I4 * RL, Vo)

eqns = [eqn1, eqn2, eqn3, eqn4, eqn5]

# Solver
Sol = sp.solve(eqns, (I1, I2, I3, I4, Vo))
si6 = Sol[I4]

# PMT
PMT = (1 / tau_1 - 1 / tau_2) / (s**2 + (1 / tau_1 + 1 / tau_2) * s + 1 / tau_1 / tau_2)

# I_out without V_in
h = RL * si6 / Vi

# Final transfer function
H1 = PMT * h

### ITERATION ##
# Associated errors of each circuit element
error = [10, 1, 1, 1,#C
             2, 2, 2,#L
               0.1,  #RL
               0, 0]  # tau1_2
error_percentual = [i/100 for i in error] # erro percentual

# number of iterations
n_iterations = 1500
all_x_coord, all_pols, y = MonteCarlo_iteration(iterations=n_iterations, erro=error_percentual, nominal_values=Cval, FT=H1, t=t1)

save = False
## PLOTS PULSE##
plot_pulso(t=t1, y, sigma=3, save_fig=save)

## PLOT POLE MAP ##
all_y_real, all_y_imag = pole_map(all_pols, save_fig=save)

## PLOT PEARSON ##
Pearson8(all_y_real, all_x_coord, pole_names_real)
Pearson8(all_y_imag, all_x_coord, pole_names_imag)

## PLOT HISTOSGRAMS ##
histogram(all_y_real, pole_names_real, save_fig=save)
histogram(all_y_imag, pole_names_imag, save_fig=save)
