import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz, iv as besseli

# -*- coding: utf-8 -*-

import numpy as np
import math
import cmath
from mpmath import besseli, exp as mp_exp

from decimal import Decimal, getcontext
from scipy.special import dawsn

getcontext().prec = 50

# === CONSTANTS ===
E = 1.6021766208e-19
C = 2.99792458e8
VP = 8.854187e-12
m_e = 9.10938215e-31
kb = 1.602176565e-19
m_h = 1.6737236e-27
mu = 1.2566370614e-6

m = m_h
B = 2.3
n_e = 6e19
n_i = 6e19
Z_i = 1
T_e = 2.3
T_i = 2.0

v_e = np.sqrt((2 * T_e * kb * 1000) / m_e)
v_i = np.sqrt((2 * T_i * kb * 1000) / m)
w_pe = np.sqrt((n_e * E ** 2) / (m_e * VP))
w_i = 140e9 * 2 * math.pi
k_i = np.sqrt(abs(w_i ** 2 - w_pe ** 2) / (C ** 2))
w_ce = -(E * B) / m_e
w_ci = (E * B) / m
debye = v_e / (math.sqrt(2) * np.sqrt((n_e * E ** 2) / (m_e * VP)))
larmor = v_e / (math.sqrt(2) * abs(w_ce))
larmor_i = v_i / (math.sqrt(2) * w_ci)

teta = 95
phi = 80
k = 2 * k_i * math.sin(math.radians(teta / 2))
sal = 1 / (k * debye)
k_perp = k * math.sin(math.radians(phi))
k_par = k * math.cos(math.radians(phi))
lambda_e = (k_perp * larmor) ** 2
lambda_i = (k_perp * larmor_i) ** 2

# === Modified Bessel Terms ===
g = Decimal(-lambda_i)
f2 = Decimal(lambda_i)
f0 = float(mp_exp(g) * besseli(0, f2))
f1 = float(mp_exp(g) * besseli(1, f2))
f4 = [0] * 200
for i in range(2, 100):
    f4[i] = float(mp_exp(g) * besseli(i, f2))

ge = Decimal(-lambda_e)
f2e = Decimal(lambda_e)
f0e = float(mp_exp(ge) * besseli(0, f2e))
f1e = float(mp_exp(ge) * besseli(1, f2e))
f4e = [0] * 200
for i in range(2, 100):
    f4e[i] = float(mp_exp(ge) * besseli(i, f2e))

# === Dielectric Tensor ===
def dielectric_tensor(w):
    alpha = (w_pe / w) ** 2
    beta = (w_ce / w) ** 2
    e1 = 1 - alpha / (1 - beta)
    e2 = alpha * math.sqrt(beta) / (1 - beta)
    e3 = 1 - alpha
    tensor = np.zeros((3, 3), dtype=complex)
    tensor[0, 0] = e1
    tensor[0, 1] = -1j * e2
    tensor[1, 0] = 1j * e2
    tensor[1, 1] = e1
    tensor[2, 2] = e3
    return tensor

def index_sqr(w, phi, mode):
    alpha = (w_pe / w) ** 2
    beta = (w_ce / w) ** 2
    gamma = (beta * math.sin(math.radians(phi)) ** 2) ** 2 + 4 * beta * (1 - alpha) ** 2 * math.cos(math.radians(phi)) ** 2
    n = 1 - (2 * alpha * (1 - alpha)) / (2 * (1 - alpha) - beta * math.sin(math.radians(phi)) ** 2 + mode * math.sqrt(gamma))
    return n

def plasma_dispersion(x):
    real = -2 * dawsn(x)
    imag = math.sqrt(math.pi) * math.exp(-x ** 2)
    return complex(real, imag)

def electron_response(w_s):
    He = complex(0, 0)
    x0 = w_s / (k_par * v_e)
    He += sal ** 2 + sal ** 2 * f0e * x0 * plasma_dispersion(x0).real
    He += 1j * sal ** 2 * f0e * x0 * plasma_dispersion(x0).imag

    x1_pos = (w_s + w_ce) / (k_par * v_e)
    x1_neg = (w_s - w_ce) / (k_par * v_e)
    He += sal ** 2 * f1e * x0 * (plasma_dispersion(x1_pos).real + plasma_dispersion(x1_neg).real)
    He += 1j * sal ** 2 * f1e * x0 * (plasma_dispersion(x1_pos).imag + plasma_dispersion(x1_neg).imag)

    for l in range(2, 100):
        x_pos = (w_s + l * w_ce) / (k_par * v_e)
        x_neg = (w_s - l * w_ce) / (k_par * v_e)
        He += sal ** 2 * f4e[l] * x0 * (plasma_dispersion(x_pos).real + plasma_dispersion(x_neg).real)
        He += 1j * sal ** 2 * f4e[l] * x0 * (plasma_dispersion(x_pos).imag + plasma_dispersion(x_neg).imag)
    return He

def ion_response(w_s):
    Hi = complex(0, 0)
    term = ((Z_i ** 2 * n_i * T_e) / (n_e * T_i))
    x0 = w_s / (k_par * v_i)
    Hi += sal ** 2 * term * f0 * x0 * plasma_dispersion(x0).real
    Hi += 1j * sal ** 2 * term * f0 * x0 * plasma_dispersion(x0).imag

    x1_pos = (w_s + w_ci) / (k_par * v_i)
    x1_neg = (w_s - w_ci) / (k_par * v_i)
    Hi += sal ** 2 * term * f1 * x0 * (plasma_dispersion(x1_pos).real + plasma_dispersion(x1_neg).real)
    Hi += 1j * sal ** 2 * term * f1 * x0 * (plasma_dispersion(x1_pos).imag + plasma_dispersion(x1_neg).imag)

    for l in range(2, 100):
        x_pos = (w_s + l * w_ci) / (k_par * v_i)
        x_neg = (w_s - l * w_ci) / (k_par * v_i)
        Hi += sal ** 2 * term * f4[l] * x0 * (plasma_dispersion(x_pos).real + plasma_dispersion(x_neg).real)
        Hi += 1j * sal ** 2 * term * f4[l] * x0 * (plasma_dispersion(x_pos).imag + plasma_dispersion(x_neg).imag)
    return Hi

def longitudinal_dielectric(w_s):
    return 1 + electron_response(w_s) + ion_response(w_s)

def electron_spectral_density(w_s):
    He = electron_response(w_s)
    epsL = longitudinal_dielectric(w_s)
    t = 1 - He / epsL
    term = abs(t) ** 2
    x0 = w_s / (k_par * v_e)
    S = 2 * (math.sqrt(math.pi) / (abs(k_par) * v_e)) * term * f0e * math.exp(-x0 ** 2)
    for l in range(1, 100):
        x_pos = (w_s + l * w_ce) / (k_par * v_e)
        x_neg = (w_s - l * w_ce) / (k_par * v_e)
        S += 2 * (math.sqrt(math.pi) / (abs(k_par) * v_e)) * term * f4e[l] * (math.exp(-x_pos ** 2) + math.exp(-x_neg ** 2))
    return S

def ion_spectral_density(w_s):
    He = electron_response(w_s)
    epsL = longitudinal_dielectric(w_s)
    t = He / epsL
    term = abs(t) ** 2
    x0 = w_s / (k_par * v_i)
    factor = 2 * (math.sqrt(math.pi) * Z_i ** 2 * n_i) / (n_e * abs(k_par) * v_i)
    S = factor * f0 * math.exp(-x0 ** 2)
    for l in range(1, 100):
        x_pos = (w_s + l * w_ci) / (k_par * v_i)
        x_neg = (w_s - l * w_ci) / (k_par * v_i)
        S += factor * f4[l] * (math.exp(-x_pos ** 2) + math.exp(-x_neg ** 2))
    return S

# Frequency sweep
f_range = np.linspace(172e9, 178e9, 200)
w_range = 2 * np.pi * f_range
spectrum = np.array([electron_spectral_density(w) for w in w_range])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(f_range * 1e-9, spectrum.real)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Scattering Function S(f)')
plt.title('Total Scattering Function vs Frequency (Stabilized)')
plt.grid(True)
plt.tight_layout()
plt.show()
