import math
import librosa
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd

def tone_estimation (f0, tone1_f0):
    ''' 
    Returns estimated/classified tone
    ** this function is built based on the rules for tone simulation 
    
    Args:
        f0 (np.array, float): time series of f0 returned from pitch extraction
            (ie. librosa.pyin)
        tone1_f0 (np.array, float): f0 of tone1, it is used for calculating the expected tone pitch values for all tones
    Returns:
        final_tone (np.array, int): tone classification results in integer (i.e. 1 represents Tone 1)'''

    fh = np.mean(tone1_f0[~np.isnan(tone1_f0)])
    f0_non_nan = f0[~np.isnan(f0)]
    
    # filter the f0 that has all NaN values
    if f0_non_nan.size != 0:
        tone = np.array([])
        # below calculation is from Table 1 of Learning tone distinctions for Mandarin Chinese paper
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
        
        # below calculation is from Table 1 of Learning tone distinctions for Mandarin Chinese paper
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
        
        # below calculation is from Table 1 of Learning tone distinctions for Mandarin Chinese paper
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
        # if the count is equal to 3, all estimated Fb, Fi and Fe values are close to the actual values
        if count[count.argmax()] == 3:
            final_tone = np.array([count.argmax()+1])
        # if the count is equal to 2, some of the estimated Fb, Fi and Fe values are close to the actual values
        else:
            final_tone = np.where(count == 2)
            final_tone = np.array([i+1 for i in final_tone])[0]
        print(count)
        print('Estimated tone: Tone ' + str(final_tone) + '\n')
    else: 
        final_tone = np.array([0])
        print('all f0 estimation is nan \n')
    return final_tone

def find_tones (exp_tones, act_value):
    ''' 
    Returns two closest estimated values compared to the actual value
    
    Args:
        exp_tones (np.array, float): expected pitch values for tone 1-4 calculated based on the tone 1 f0
        act_value (np.array, float): actual value from the recordings 
    Returns:
        final_tones (np.array, int): selected two values from exp_tones that are closest to act_value'''

    # copy of exp_tones
    list = np.array(copy.deepcopy(exp_tones))
    # find a value's index that has the smallest difference
    idx1 = np.abs(list-act_value).argmin()
    est1 = list[idx1]
    list = np.delete(list, idx1, 0)
    # find a value's index that has the smallest difference in the updated list
    idx2 = np.abs(list-act_value).argmin()
    est2 = list[idx2]

    return np.array([exp_tones.index(est1), exp_tones.index(est2)])

def tone_classification (algo, speaker):
    ''' 
    Returns tone classification results based on the algorithm to use and speaker type
    
    Args:
        algo (str, parselmouth or librosa): used algorithm to get f0 estimation
        speaker (str): speaker type (i.e. FV1 from Tone Perfect corpus)'''

    print('used f0 estimation algo: ' + algo + ' and speaker: ' + speaker)
    # based on the input algorithm, select the pre-parsed pitch data accordingly 
    if (algo == 'parselmouth'):
        pitch = pd.read_json('../data/parsed/toneperfect_pitch_parselmouth.json')
    elif (algo == 'librosa'):
        pitch = pd.read_json('../data/parsed/toneperfect_pitch_librosa_50-500-fminmax.json')

    # select data that is only related to the speaker specified by the arg 'speaker'
    data = pitch.loc[pitch['speaker'] == speaker]
    classification_result = np.array([], dtype=np.int64)
    
    for d in np.array(data):
        # d[7] is the pitch_contour data
        f0 = np.array(d[7], dtype=np.float64)
        # remove all None data
        f0 = f0[f0 != np.array(None)]
        print('Syllable: ' + d[2] + ' and tone ' + str(d[3]) + '\n')
        
        # d[3] is the tone data
        if d[3] == 1:
            tone = tone_estimation(f0, f0)
        else:
            # d[2] is the syllable data
            # find the matching syllable's Tone 1 pitch_contour data
            tone1 = np.array(data.loc[(data['tone'] == 1) & (data['syllable'] == d[2])])
            tone1_f0 = np.array(tone1[0][7], dtype=np.float64)
            tone1_f0 = tone1_f0[tone1_f0 != np.array(None)]
            tone = tone_estimation(f0, tone1_f0)
        
        # if the actual tone is in the classified result and the classified result has only one value, it is correctly classified
        if ((d[3] in tone) & (tone.size == 1)):
            classification_result = np.append(classification_result, 1)
        # if the actual tone is in the classified result and the classified result has more than one values, it is semi-correctly classified
        elif ((d[3] in tone) & (tone.size > 1)):
            classification_result = np.append(classification_result, 2)
        # else, it is incorrectly classified
        else:
            classification_result = np.append(classification_result, 0)
            print('incorrect: '+str(tone))

    print('classification result\n')
    count = np.bincount(classification_result)
    print('total: '+str(classification_result.size))
    print('correct: '+str(count[1]))
    print('semi-correct: '+str(count[2]))
    print('incorrect: '+str(count[0]))
    pass
