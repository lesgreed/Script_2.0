import numpy as np
from scipy.constants import epsilon_0, c, e, pi
from scipy.special import iv as besseli
from scipy.special import wofz  # Plasma dispersion function
import matplotlib.pyplot as plt

# === Constants ===
B0 = 2.3  # Tesla
f0 = 174e9  # Hz
theta_s = 120  # degrees
theta_kB = 98  # degrees
polarization = 'X'  # 'X' or 'O'

# === Species Data ===
# Format: [mass, charge (e), density, temperature (eV), drift velocity]
species = np.array([
    [9.10938356e-31, -1, 1e19, 1000, 0],  # Electrons
    [1.6726219e-27,   1, 1e19, 1000, 0],  # Protons
    [3.344e-27,       1, 5e18, 1000, 0],  # Deuterium ions
    [6.644e-27,       1, 2e18, 1000, 0]   # Helium ions
])

# === Frequency range ===
f = np.arange(-f0 - 3e8, f0 + 3e8, 1e7)
w = 2 * pi * f
c = 3*10**8
omega0 = 2 * pi * f0
k0 = omega0 / c
k_scat = 2 * k0 * np.sin(np.deg2rad(theta_s / 2))
N_species = species.shape[0]
Lmax = 10
l = np.arange(-Lmax, Lmax + 1)

# === Helper Functions ===

def debye_length(T, Z, n):
    return np.sqrt(epsilon_0 * T * e / (Z * n * e**2))

def cyclotron_frequency(q, m, B):
    return np.abs(q * B / m)

def disp_func(xi):
    return 1j * np.sqrt(pi) * wofz(xi)[0]

# === Derived quantities ===
lambda_D = debye_length(species[0, 3], 1, species[0, 2])
alpha = 1 / (k_scat * lambda_D)
S = np.zeros((len(w), N_species))

# === Main Calculation Loop ===
for k_idx, wk in enumerate(w):
    H = np.zeros(N_species, dtype=np.complex128)
    for i in range(N_species):
        m, Z, n, T, v_drift = species[i]
        q = Z * e
        v_th = np.sqrt(2*T * e / m)
        omega_c = abs(cyclotron_frequency(q, m, B0))
        k_par = k_scat * np.cos(np.deg2rad(theta_kB))
        k_perp = k_scat * np.sin(np.deg2rad(theta_kB))
        rho = v_th / omega_c

        tmp2 = besseli(l, (k_perp * rho)**2) 
        tmp2 *= np.exp(-(k_perp*rho)**2)

        xi = (wk - l * omega_c) / (k_par * v_th)
        tmp2 = tmp2.astype(np.complex128)
        tmp2 *= (1 + (wk * disp_func(xi) / (k_par * v_th)))
        


        if i == 0:
            H[i] = alpha**2 * np.exp(-(k_perp * rho)**2) * np.sum(tmp2)
        else:
            scale = alpha**2 * Z**2 * n * species[0,3] / (species[0,2] * T)
            H[i] = scale * np.exp(-(k_perp * rho)**2) * np.sum(tmp2)

    eps_L = 1 + np.sum(H)

    # Compute scattering functions
    for i in range(N_species):
        m, Z, n, T, v_drift = species[i]
        q = Z * e
        v_th = np.sqrt(2*T * e / m)
        omega_c = cyclotron_frequency(q, m, B0)
        k_par = k_scat * np.cos(np.deg2rad(theta_kB))
        k_perp = k_scat * np.sin(np.deg2rad(theta_kB))
        rho = abs(v_th / omega_c)

        tmp2 = besseli(l, (k_perp * rho)**2)
        xi = (wk - l * omega_c) / (k_par * v_th)
        gauss = np.exp(-xi**2)
        if i == 0:  # Electrons
            S[k_idx, i] = abs(1 - H[i] / eps_L)**2 * 2 * np.sqrt(pi) / abs(k_par) / v_th
            S[k_idx, i] *= np.exp(-(k_perp * rho)**2)
            S[k_idx, i] *= np.sum(gauss)
        else:
            scale = abs(H[0] / eps_L)**2 * 2 * np.sqrt(pi) * Z**2 * n / species[i, 2] / v_th / abs(k_par)
            S[k_idx, i] = scale * np.exp(-(k_perp * rho)**2) * np.sum(tmp2*gauss)
S_total = np.sum(S, axis=1)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(f * 1e-9, S_total, label='Total Scattering Function S(f)')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Scattering Function S(f)')
plt.title('Total Scattering Function vs Frequency')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()