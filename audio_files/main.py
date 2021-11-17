from matplotlib import pyplot as plt
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
from scipy.io.wavfile import read
from scipy.fftpack import fft


def read_data():
    f2, B = read('Acris_crepitans85.wav')
    f1, A = read('Acris_blanchardi29.wav')
    f3, C = read('Acris_gryllus85.wav')
    f4, D = read('Adenomera_bokermanni1.wav')
    f5, E = read('Adenomera_coca.wav')
    f6, F = read('Adenomera_cotuba95.wav')
    f7, G = read('Adenomera_juikitam82.wav')
    f8, H = read('Ademomera_marmorata.wav')
    f9, I = read('Adenomera_martinezi20.wav')
    f10, J = read('Adenomera_saci93.wav')
    f11, K = read('Adenomera_thomei.wav')
    f12, L = read('Afrixalus_dorsalis.wav')
    f13, M = read('Afrixalus_dorsalis.wav')
    f14, N = read('Afrixalus_weidholzi.wav')
    f15, O = read('Aglyptodactylus_laticeps.wav')
    f16, P = read('Aglyptodactylus_madagascariensis.wav')
    f17, Q = read('Anaxyrus_retiformis.wav')
    f18, R = read('Anaxyrus_nelsoni.wav')
    f19, S = read('Anaxyrus_punctatus1.wav')
    f19, S = read('Anaxyrus_punctatus2.wav')
    f20, T = read('Anaxyrus_debilis.wav')
    f21, U = read('Anaxyrus_canorus1.wav')
    f22, V = read('Anaxyrus_boreas_boreas_2.wav')
    f23, W = read('Anaxyrus_fowleri52.wav')
    f24, X = read('Anaxyrus_microscaphus.wav')
    f25, Y = read('Anaxyrus_boreas_halophilus1.wav')
    f26, Z = read('Anaxyrus_exsul1.wav')
    data = np.array([A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z])
    return data

def run_main():
    data = read_data()
    data = fft_data_array(data)
    input1, input = read('Anaxyrus_punctatus2.wav')
    for x in range(np.shape(query_QF)[0]):
        new_data = data_NF[np.argsort(np.linalg.norm(data[x] - input , ord = 2, axis = 1))]
    fft_out = fft(input)

def fft_data_array(x):
    Transformed_array = np.empty(np.shape(x))
    for i in range(26):
        Transformed_array[i] = fft(x[i])

    return fft_data_array




run_main()
