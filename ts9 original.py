import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio
import matplotlib.patches as patches
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
#from pytc2.sistemas_lineales import plot_plantilla

#filtro normalizado -> todas las singularidades en el circulo unitario?
#--- Plantilla de diseño ---

fs = 1000
wp = [0.8, 35] #freq de corte/paso (rad/s)
ws = [0.1, 40] #freq de stop/detenida (rad/s)

#si alpha_p es =3 -> max atenuacion, butter

alpha_p = 1/2 #atenuacion de corte/paso, alfa_max, perdida en banda de paso 
alpha_s = 40/2 #atenuacion de stop/detenida, alfa_min, minima atenuacion requerida en banda de paso 

#Aprox de modulo

#
#
#f_aprox = 'ellip'
#

#Aprox fase
#f_aprox = 'bessel'

# --- Diseño de filtro analogico ---
f_aprox = 'butter'
mi_sos_butter = signal.iirdesign(wp = wp, ws = ws, gpass = alpha_p, gstop = alpha_s, analog = False, ftype = f_aprox, output ='sos', fs=fs) #devuelve dos listas de coeficientes, b para P y a para Q
# f_aprox = 'cheby1'
# mi_sos_cheby1 = signal.iirdesign(wp = wp, ws = ws, gpass = alpha_p, gstop = alpha_s, analog = False, ftype = f_aprox, output ='sos', fs=fs) #devuelve dos listas de coeficientes, b para P y a para Q
# f_aprox = 'cheby2'
# mi_sos_cheby2 = signal.iirdesign(wp = wp, ws = ws, gpass = alpha_p, gstop = alpha_s, analog = False, ftype = f_aprox, output ='sos', fs=fs) #devuelve dos listas de coeficientes, b para P y a para Q
f_aprox = 'cauer'
mi_sos_cauer = signal.iirdesign(wp = wp, ws = ws, gpass = alpha_p, gstop = alpha_s, analog = False, ftype = f_aprox, output ='sos', fs=fs) #devuelve dos listas de coeficientes, b para P y a para Q

# %%
mi_sos = mi_sos_cauer

# --- Respuesta en frecuencia ---
w, h= signal.freqz_sos(mi_sos, worN = np.logspace(-2, 1.9, 1000), fs = fs) #calcula rta en frq del filtro, devuelve w y vector de salida (h es numero complejo)

# --- Cálculo de fase y retardo de grupo ---

fase = np.unwrap(np.angle(h)) #unwrap hace grafico continuo

w_rad = w / (fs / 2) * np.pi
gd = -np.diff(fase) / np.diff(w_rad) #retardo de grupo [rad/rad]

# --- Polos y ceros ---

z, p, k = signal.sos2zpk(mi_sos) #ubicacion de polos y ceros, z=ubicacion de ceros(=0), p=ubicacion polos, k

# --- Gráficas ---
#plt.figure(figsize=(12,10))

# Magnitud
plt.subplot(2,2,1)
plt.plot(w, 20*np.log10(abs(h)), label = f_aprox)
plt.title('Respuesta en Magnitud')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')

# Fase
plt.subplot(2,2,2)
plt.plot(w, fase, label = f_aprox)
plt.title('Fase')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')

# Retardo de grupo
plt.subplot(2,2,3)
plt.plot(w[:-1], gd, label = f_aprox)
plt.title('Retardo de Grupo ')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('τg [# muestras]')
plt.grid(True, which='both', ls=':')

# Diagrama de polos y ceros
plt.subplot(2,2,4)
plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label=f'{f_aprox} Polos')
if len(z) > 0:
    plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'{f_aprox} Ceros')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title('Diagrama de Polos y Ceros (plano z)')
plt.xlabel('σ [rad/s]')
plt.ylabel('jω [rad/s]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#%%
#FILTROS IIR FIR DISIGN
fs = 1000
wp = [0.8, 35] #freq de corte/paso (rad/s)
ws = [0.1, 35.7] #freq de stop/detenida (rad/s)

f_aprox = 'butter'

frecuencias=np.sort(np.concatenate(((0, fs/2), wp, ws)))
deseado=[0, 0, 1, 1, 0, 0]
cant_coef= 2000
retardo=(cant_coef-1)//2

#fir_win_rect=signal.firwin2(numtaps=cant_coef, fs=fs , freq= frecuencias, gain= deseado,  nfreqs=int((np.ceil(np.sqrt(cant_coef))2)*2) -1, window='boxcar' ) #te devuelve filtro factorizado.
fir_win_rect = signal.firwin2(
    numtaps=cant_coef,
    fs=fs,
    freq=frecuencias,
    gain=deseado,
    nfreqs=int((np.ceil(np.sqrt(cant_coef)) * 2)**2) - 1,
    window='boxcar'
)


cant_coef= 4001
retardo=(cant_coef-1)//2
weight_ls= [0.25, 0.5, 0.25]
fir_win_ls=signal.firls(numtaps=cant_coef, fs=fs , bands= frecuencias, desired= deseado, weight=weight_ls) #te devuelve filtro factorizado.

cant_coef= 3001
retardo=(cant_coef-1)//2
deseado=[0, 1, 0]
weight_remez= [1, 0.5, 1]
fir_win_pm=signal.remez(numtaps=cant_coef, fs=fs , bands= frecuencias, desired= deseado, weight= weight_remez ) #te devuelve filtro factorizado.


#w1, h1 = signal.freqz(b=fir_win_rect, worN=np.logspace(-2, 1.9, 1000), fs=fs)
w2, h2 = signal.freqz(b=fir_win_ls, worN=np.logspace(-2, 1.9, 3000), fs=fs)
#w3, h3 = signal.freqz(b=fir_win_pm, worN=np.logspace(-2, 1.9, 3000), fs=fs) #equiripple en banda de stop y de paso.



# --- Cálculo de fase y retardo de grupo ---

fase = np.unwrap(np.angle(h)) #unwrap hace grafico continuo
w_rad = w / (fs / 2) * np.pi
#retardo en grupo= -dfase/dw
gd = -np.diff(fase) / np.diff(w_rad) #retardo de grupo [rad/rad]

# --- Polos y ceros ---

#z, p, k = signal.sos2zpk(signal.tf2sos(b=fit_wing_hamming, a=1)) #ubicacion de polos y ceros, z=ubicacion de ceros(=0), p=ubicacion polos, k

# --- Gráficas ---
plt.figure(figsize=(12,10))

# Magnitud
plt.subplot(3,1,1)
plt.plot(w2, 20*np.log10(abs(h2)), label =f_aprox)
#plot_plantilla(filter_type= 'bandpass', fpass=wp, ripple= alpha_p*2, fstop= ws , attenuation= alpha_s*2 , fs=fs )
plt.title('Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')

# Fase
plt.subplot(3,1,2)
plt.plot(w, fase, label = f_aprox)
plt.title('Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')

# Retardo de grupo
plt.subplot(3,1,3)
plt.plot(w[:-1], gd, label = f_aprox)
plt.title('Retardo de Grupo ')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [# muestras]')
plt.grid(True, which='both', ls=':')


# # Diagrama de polos y ceros
# plt.figure(figsize=(10,10))
# plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label='{f_aprox} Polos')
# axes_hdl=plt.gca()

# if len(z)>0:
#     plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label='{f_aprox} Polos')
# plt.axhline(0, color='k', lw=0.5)
# plt.axvline(0, color='k', lw=0.5)
# unit_circle = patches.Circle((0,0), radius=1, fill=False, color='gray', ls='dotted', lw=2)
# axes_hdl.add_patch(unit_circle)

# plt.axis([-1.1,1.1, -1.1, 1.1])
# plt.title('Diagrama de Polos y Ceros (plano Z)')
# plt.xlabel(r'$|Re(z)$')
# plt.ylabel(r'$|Im(z)$')
# plt.legend()
# plt.grid(True)
# plt.legend()

# plt.tight_layout()
# plt.show()

#PARA VER 40 DB EN MÓDULO
#-No sirvió aumentar la grilla (lo ult en fir_win_hamming)
#fit_wing_hamming=signal.firwin2(numtaps=cant_coef, fs=fs , freq= frecuencias, gain= deseado, nfreqs=int((np.ceil(np.sqrt(cant_coef))2)*2) -1, ) #te devuelve filtro factorizado.
#resolvimos cambiando la ventana por la rectangular



#%%

##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
#io.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()


N = len(ecg_one_lead)

ecg_filt_butt = signal.sosfiltfilt(mi_sos_butter, ecg_one_lead)
ecg_filt_cauer = signal.sosfiltfilt(mi_sos_cauer, ecg_one_lead)
# ecg_filt_cheb1 = signal.sosfiltfilt(mi_sos_cheby1, ecg_one_lead)
# ecg_filt_cheb2 = signal.sosfiltfilt(mi_sos_cheby2, ecg_one_lead)


ecg_filt_win= signal.lfilter(b= fir_win_rect, a=1, x=ecg_one_lead )


#%%
#plt.figure()

#plt.plot(ecg_one_lead, label = 'ecg raw')
# plt.plot(ecg_filt_butt, label = 'butter')
#plt.plot(ecg_filt_cauer, label = 'cauer')
#plt.plot(ecg_filt_cheb1, label = 'cheby1')
#plt.plot(ecg_filt_cheb2, label = 'cheby2')

plt.legend()

# hb_1 = mat_struct['heartbeat_pattern1']
# hb_2 = mat_struct['heartbeat_pattern2']

# plt.figure()
# plt.plot(ecg_one_lead[5000:12000])

# plt.figure()
# plt.plot(hb_1)

# plt.figure()
# plt.plot(hb_2)

##################
## ECG sin ruido
##################

#################################
# Regiones de interés sin ruido #
#################################

cant_muestras = len(ecg_one_lead)

regs_interes = (
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ecg_filt_butt[zoom_region], label='Butterworth')
    plt.plot(zoom_region, ecg_filt_win[zoom_region + retardo], label='FIR Window')
   
    plt.title('ECG sin ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
 
#################################
# Regiones de interés con ruido #
#################################
 
regs_interes = (
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    resul1_mediana=signal.medfilt(ecg_one_lead[zoom_region], kernel_size=201)#en fir tiene que ser impar para tener retardo entero
    estimacion=signal.medfilt(resul1_mediana, kernel_size=601)
    
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ecg_filt_butt[zoom_region], label='Butterworth')
    plt.plot(zoom_region, ecg_filt_win[zoom_region + retardo], label='FIR Window')
    plt.plot(zoom_region, estimacion, label='Estimacion')
   
    plt.title('ECG con ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
    
    plt.figure()
    plt.plot(zoom_region,estimacion)
    plt.show()

    
           
    plt.show()

# %% Interpolacion --> metodo estimacion sustraccion
qrs_detections = mat_struct['qrs_detections'].flatten()
n0 = int(0.08 * fs)  # 80 ms antes del QRS


# Construir puntos S
m_points = qrs_detections - n0
m_points = m_points[m_points > 0]  # evitar índices negativos
s_values = ecg_one_lead[m_points] 

# Interpolación spline cúbica
cs = CubicSpline(m_points, s_values)
baseline = cs(np.arange(len(ecg_one_lead)))

# ECG corregido (remover línea de base)
ecg_corrected = ecg_one_lead - baseline

# Crear una sola figura con subplots
fig, axes = plt.subplots(len(regs_interes), 1, figsize=(14, 10), sharex=False)


# Título general para toda la figura
fig.suptitle("Spline cubico", fontsize=16)

for idx, reg in enumerate(regs_interes):
    start, end = reg.astype(int)
    ecg_region = ecg_one_lead[start:end]
    t_region = np.arange(start, end) / fs

    # Filtrar QRS en la región
    qrs_region = qrs_detections[(qrs_detections >= start) & (qrs_detections < end)]

    # Construir puntos PQ
    m_points = qrs_region - n0
    m_points = m_points[m_points > start]
    s_values = ecg_one_lead[m_points]

    # Interpolación spline cúbica
    cs = CubicSpline(m_points, s_values)
    baseline_region = cs(np.arange(start, end))

    # ECG corregido
    ecg_corrected = ecg_region - baseline_region
    
    
    # Agregar cruces verdes en QRS detectados
    axes[idx].plot(t_region[qrs_region - start], ecg_region[qrs_region - start], 'bx', label='QRS detecciones')
    
    axes[idx].plot(t_region[m_points - start], ecg_region[m_points - start], 'go', label='n₀ (80 ms antes)')


    # Graficar en el subplot correspondiente
    axes[idx].plot(t_region, ecg_region, label='ECG original', alpha=0.7)
    axes[idx].plot(t_region, baseline_region, label='Línea de base', linestyle='--')
    axes[idx].plot(t_region, ecg_corrected, label='ECG corregido', alpha=0.8)
    axes[idx].set_title(f'Región {idx+1}: {start/fs/60:.1f}-{end/fs/60:.1f} min')
    axes[idx].set_xlabel('Tiempo [s]')
    axes[idx].set_ylabel('Amplitud')
    axes[idx].grid(True)
    axes[idx].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajustar para que no tape el título general
plt.show()


# %% Para todo el ecg
qrs_detections = mat_struct['qrs_detections'].flatten()
n0 = int(0.08 * fs)  # 80 ms antes del QRS


# Construir puntos S
m_points = qrs_detections - n0
m_points = m_points[m_points > 0]  # evitar índices negativos
s_values = ecg_one_lead[m_points] 

# Interpolación spline cúbica
cs = CubicSpline(m_points, s_values)
baseline = cs(np.arange(len(ecg_one_lead)))

# ECG corregido (remover línea de base)
ecg_corrected = ecg_one_lead - baseline

# Vector de tiempo
t = np.arange(len(ecg_one_lead)) / fs

# Graficar
plt.figure(figsize=(14, 6))
#plt.plot(t, ecg_one_lead, label='ECG original (con ruido)', alpha=0.7)
plt.plot(t, baseline, label='Línea de base estimada', linestyle='--')
plt.plot(t, ecg_corrected, label='ECG corregido', alpha=0.8)

# QRS detections (cruces azules)
plt.plot(t[qrs_detections], ecg_one_lead[qrs_detections], 'bx', label='QRS detecciones')

# m_points (círculos verdes)
m_points = qrs_detections - n0
m_points = m_points[m_points > 0]  # evitar negativos
plt.plot(t[m_points], ecg_one_lead[m_points], 'go', label='n₀ (80 ms antes)')

# Configuración
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('ECG con corrección de línea de base y referencias QRS/n₀')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% Matched filter


# ============================
# ENTRADAS: usa tus datos reales
# ============================
# ecg_one_lead: señal ECG (array 1D)
# qrs_pattern1: plantilla QRS (array 1D)
# qrs_detections: índices de latidos reales (array 1D)

# Ejemplo: reemplaza estas líneas por tus datos
# ecg_one_lead = ...
# qrs_pattern1 = ...
# qrs_detections = ...

# ============================
# 1. Crear filtro adaptado
# ============================

patron= mat_struct['qrs_pattern1'].flatten()
patron_2 = patron - np.mean(patron) #para tener area neta nula, util para filtrar
# ============================
# 2. Correlación (convolución)
# ============================
ecg_detection = signal.lfilter(b=patron_2, a=1, x=ecg_one_lead)
ecg_detection_abs = np.abs(ecg_detection)
ecg_detection_abs = ecg_detection_abs/np.std(ecg_detection_abs)
# ============================
# 3. Detectar picos con find_peaks
# ============================
threshold = 0.5 * np.max(ecg_detection_abs)  # umbral simple
peaks, properties = find_peaks(ecg_detection_abs, height=threshold, distance=100)

# ============================
# 4. Comparar con ground truth
# ============================
TP = sum([any(abs(p - qrs_detections) <= 20) for p in peaks])
FP = len(peaks) - TP
FN = len(qrs_detections) - TP
sensibilidad = TP / (TP + FN)
# ppv = TP / (TP + FP)

print("Resultados del detector:")
print(f"Número de picos detectados: {len(peaks)}")
# print(f"TP: {TP}, FP: {FP}, FN: {FN}")
# print(f"Sensibilidad: {sensibilidad:.2f}, PPV: {ppv:.2f}")

# ============================
# 5. Graficar resultados
# ============================
plt.figure(figsize=(12, 6))

# Señal ECG con detecciones
plt.subplot(2, 1, 1)
plt.plot(ecg_one_lead, label='ECG')
plt.scatter(qrs_detections, ecg_one_lead[qrs_detections], color='green', label='Ground Truth')
plt.scatter(peaks, ecg_one_lead[peaks], color='red', label='Detectados')
plt.title('Señal ECG con detecciones')
plt.legend()

# Salida del filtro adaptado
plt.subplot(2, 1, 2)
plt.plot(ecg_detection, label='Correlación (Filtro adaptado)')
plt.scatter(peaks, ecg_detection[peaks], color='red', label='Picos detectados')
plt.title('Salida del filtro adaptado')
plt.legend()

plt.tight_layout()
plt.show()





