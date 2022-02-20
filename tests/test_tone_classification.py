#%%
import math
import librosa
import numpy as np
import matplotlib.pyplot as plt
import copy

y1_f1, sr1_f1 = librosa.load("../tone_perfect/a1_FV1_MP3.mp3")
y2_f1, sr2_f1 = librosa.load("../tone_perfect/a2_FV1_MP3.mp3")
y3_f1, sr3_f1 = librosa.load("../tone_perfect/a3_FV1_MP3.mp3")
y4_f1, sr4_f1 = librosa.load("../tone_perfect/a4_FV1_MP3.mp3")
f0_1, voiced_flag, voiced_probs = librosa.pyin(y1_f1, fmin=50, fmax=500)
f0_2, voiced_flag, voiced_probs = librosa.pyin(y2_f1, fmin=50, fmax=500)
f0_3, voiced_flag, voiced_probs = librosa.pyin(y3_f1, fmin=50, fmax=500)
f0_4, voiced_flag, voiced_probs = librosa.pyin(y4_f1, fmin=50, fmax=500)

fh = np.mean(f0_1[1:])
f1 = f0_1[~np.isnan(f0_1)]
f2 = f0_2[~np.isnan(f0_2)]
f3 = f0_3[~np.isnan(f0_3)]
f4 = f0_4[~np.isnan(f0_4)]

plt.plot(f1)
plt.plot(f2)
plt.plot(f3)
plt.plot(f4)
plt.show()

def tone_estimation (f0, tone1_f0):
    ''' 
    Returns estimated/classified tone
    ** this function is built based on the rules for tone simulation 
    
    Args:
        f0: time series of f0 returned from pitch extraction
            (ie. librosa.pyin)
        tone1_f0: f0 of tone1, it is used for calculating the expected tone pitch values for all tones
    Returns:
        final_tone: tone classificationx result in integer'''

    fh = np.mean(tone1_f0[~np.isnan(tone1_f0)])
    f0_non_nan = f0[~np.isnan(f0)]

    tone = np.array([])
    exp_T1_Fb = fh
    exp_T2_Fb = fh*np.sqrt(0.5)-fh/2*1/12
    exp_T3_Fb = fh*np.sqrt(0.5)
    exp_T4_Fb = fh+fh/2*2/12
    act_f0_b = f0_non_nan[1]
    exp_tone_Fb = [exp_T1_Fb, exp_T2_Fb, exp_T3_Fb, exp_T4_Fb]
    tone = np.concatenate((tone, find_tones(exp_tone_Fb, act_f0_b)))
    
    # print('Expected T1 beginning freq: ', exp_T1_Fb)
    # print('Expected T2 beginning freq: ', exp_T2_Fb)
    # print('Expected T3 beginning freq: ', exp_T3_Fb)
    # print('Expected T4 beginning freq: ', exp_T4_Fb)
    # print('Actual f0 beginning value: ', act_f0_b, '\n')
    
    exp_T1_Fe = exp_T1_Fb
    exp_T2_Fe = fh+fh/2*2/12
    exp_T3_Fe = exp_T3_Fb
    exp_T4_Fe = 0.5*(fh+fh/2*2/12)
    act_f0_e = f0_non_nan[-1]
    exp_tone_Fe = [exp_T1_Fe, exp_T2_Fe, exp_T3_Fe, exp_T4_Fe]
    tone = np.concatenate((tone, find_tones(exp_tone_Fe, act_f0_e)))

    # print('Expected T1 end freq: ', exp_T1_Fe)
    # print('Expected T2 end freq: ', exp_T2_Fe)
    # print('Expected T3 end freq: ', exp_T3_Fe)
    # print('Expected T4 end freq: ', exp_T4_Fe)
    # print('Actual f0 end value: ', act_f0_e, '\n')
    
    exp_T1_Fi = (exp_T1_Fb + exp_T1_Fe)/2
    exp_T2_Fi = (exp_T2_Fb + exp_T2_Fe)/2
    exp_T3_Fi = fh*0.5 - fh/2*3/12
    exp_T4_Fi = (exp_T4_Fb + exp_T4_Fe)/2
    act_f0_i = f0_non_nan[len(f0_non_nan)//2]
    exp_tone_Fi = [exp_T1_Fi, exp_T2_Fi, exp_T3_Fi, exp_T4_Fi]
    tone = np.concatenate((tone,find_tones(exp_tone_Fi, act_f0_i)))

    # print('Expected T1 middle freq: ', exp_T1_Fi)
    # print('Expected T2 middle freq: ', exp_T2_Fi)
    # print('Expected T3 middle freq: ', exp_T3_Fi)
    # print('Expected T4 middle freq: ', exp_T4_Fi)
    # print('Actual f0 middle value: ', act_f0_i, '\n')
    
    tone = tone.astype(int)
    count = np.bincount(tone)
    final_tone = count.argmax()+1
    print(count)
    print('Estimated tone: Tone ' + str(final_tone))
    return final_tone

def find_tones (exp_tones, act_value):
    list = copy.deepcopy(exp_tones)
    est1 = find_nearest(list, act_value)
    list.remove(est1)
    est2 = find_nearest(list, act_value)

    return np.array([exp_tones.index(est1), exp_tones.index(est2)])

def find_nearest (array, value):
    array = np.asarray(array)
    idx = np.abs(array-value).argmin()
    return array[idx]

# tone_estimation(f0_1, f0_1)
# tone_estimation(f0_2, f0_1)
# tone_estimation(f0_3, f0_1)
# tone_estimation(f0_4, f0_1)
# %%
