import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium

from math import radians, cos, sin, asin, sqrt
from scipy.signal import butter, filtfilt
from streamlit_folium import st_folium

# DATA
df_gps = pd.read_csv("https://raw.githubusercontent.com/eemildev/fysiikan_loppuprojekti/main/Location.csv")
df_acc = pd.read_csv("https://raw.githubusercontent.com/eemildev/fysiikan_loppuprojekti/main/Linear_Accelerometer.csv")

st.title("Päivän liikunta")

# HAVERSINE
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c  # km

# FOURIER-ANALYYSI
signal = df_acc["Y (m/s^2)"]
t = df_acc["Time (s)"]

dt = np.mean(np.diff(t))
N = len(signal)

fft = np.fft.fft(signal)
psd = np.abs(fft)**2 / N
freq = np.fft.fftfreq(N, dt)

mask = freq > 0
f_max = freq[mask][np.argmax(psd[mask])]

fourier_analyysi_askelmäärä = int(
    round(f_max * (t.iloc[-1] - t.iloc[0]))
)

# SUODATUS + ASKELMÄÄRÄ
data = signal
dt_acc = dt
fs = 1 / dt_acc
nyq = fs / 2

cutoff = 2.5  # Hz
order = 3

b, a = butter(order, cutoff / nyq, btype="low")
data_filt = filtfilt(b, a, data)

suodatettu_askelmäärä = 0
for i in range(len(data_filt) - 1):
    if data_filt[i] * data_filt[i + 1] < 0:
        suodatettu_askelmäärä += 0.5

# GPS-MATKA
kokonaismatka_km = 0.0
for i in range(len(df_gps) - 1):
    kokonaismatka_km += haversine(
        df_gps["Longitude (°)"].iloc[i],
        df_gps["Latitude (°)"].iloc[i],
        df_gps["Longitude (°)"].iloc[i + 1],
        df_gps["Latitude (°)"].iloc[i + 1],
    )

kokonaismatka = kokonaismatka_km * 1000  # m
kokonaisaika = df_gps["Time (s)"].iloc[-1] - df_gps["Time (s)"].iloc[0]

keskinopeus = kokonaismatka / kokonaisaika if kokonaisaika > 0 else 0

# ASKELPITUUS
askelpituus_fourier = (
    kokonaismatka / fourier_analyysi_askelmäärä
    if fourier_analyysi_askelmäärä > 0 else 0
)

askelpituus_suodatus = (
    kokonaismatka / suodatettu_askelmäärä
    if suodatettu_askelmäärä > 0 else 0
)

# TULOSTUS
st.subheader("Tulokset")
st.write(f"Askelmäärä laskettuna suodatuksen avulla: {suodatettu_askelmäärä:.0f}")
st.write(f"Askelmäärä laskettuna Fourier-analyysin avulla: {fourier_analyysi_askelmäärä}")
st.write(f"Kokonaismatka: {kokonaismatka:.1f} m")
st.write(f"Keskinopeus: {keskinopeus:.2f} m/s")
st.write(f"Askelpituus laskettuna suodatuksen avulla: {askelpituus_suodatus:.2f} m")
st.write(f"Askelpituus laskettuna Fourier-analyysin avulla: {askelpituus_fourier:.2f} m")

# KUVAT
st.subheader("Suodatettu kiihtyvyys (Y-akseli)")
plt.figure(figsize=(12, 4))
plt.plot(t, data_filt)
plt.xlabel("Aika (s)")
plt.ylabel("Kiihtyvyys (m/s²)")
plt.grid()
st.pyplot(plt)

st.subheader("Tehospektri")
plt.figure(figsize=(8, 4))
plt.plot(freq[mask], psd[mask])
plt.xlabel("Taajuus (Hz)")
plt.ylabel("Teho")
plt.grid()
st.pyplot(plt)

# KARTTA
st.subheader("Karttakuva")
start_lat = df_gps["Latitude (°)"].mean()
start_lon = df_gps["Longitude (°)"].mean()

m = folium.Map(location=[start_lat, start_lon], zoom_start=14)
folium.PolyLine(
    df_gps[["Latitude (°)", "Longitude (°)"]],
    color="red",
    weight=2.5
).add_to(m)

st_folium(m, width=900, height=650)
