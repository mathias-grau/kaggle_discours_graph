import pandas as pd
from pathlib import Path
import numpy as np
import json
import tqdm 

path_to_training = Path("training")
path_to_test = Path("test")

def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]

training_set = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2012', 'ES2013', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002', 'IS1003', 'IS1004', 'IS1005', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010', 'TS3011', 'TS3012']
training_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in training_set])
training_set.remove('IS1002a')
training_set.remove('IS1005d')
training_set.remove('TS3012c')

test_set = ['ES2003', 'ES2004', 'ES2011', 'ES2014', 'IS1008', 'IS1009', 'TS3003', 'TS3004', 'TS3006', 'TS3007']
test_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in test_set])

def create_train_transcriptions_df():
    lst_transcription_df = []
    for transcription_id in training_set:
        with open(path_to_training / f"{transcription_id}.json", "r") as file:
            transcription = json.load(file)
            transcription_df = pd.DataFrame(transcription, columns=['speaker','text','index'])
            transcription_df['transcription_id'] = transcription_id
        lst_transcription_df.append(transcription_df)
    transcriptions_df = pd.concat(lst_transcription_df)
    transcriptions_df.to_csv('train_transcriptions_df.csv', index=False)

def create_train_correspondances_df():
    lst_corresondances_df = []
    for transcription_id in training_set:
        with open(path_to_training / f"{transcription_id}.txt", "r") as file:
            correspondance = np.loadtxt(file,delimiter = ' ', dtype = str)
            correspondance_df = pd.DataFrame(correspondance,columns=['1','type','2'])
            correspondance_df['transcription_id'] = transcription_id
        lst_corresondances_df.append(correspondance_df)
    correspondances_df = pd.concat(lst_corresondances_df)
    correspondances_df.to_csv('train_correspondances_df.csv', index=False)

def create_training_labels_df():
    with open("training_labels.json", "r") as file:
        training_labels = json.load(file)
    lst_labels_df = []
    for transcription_id in training_set:
        label_df = pd.DataFrame(training_labels[transcription_id], columns=['label'])
        label_df['transcription_id'] = transcription_id
        label_df['index']=label_df.index
        lst_labels_df.append(label_df)
    training_labels_df = pd.concat(lst_labels_df)
    training_labels_df.to_csv('training_labels_df.csv', index=False)

def create_test_transcriptions_df():
    lst_transcription_df = []
    for transcription_id in test_set:
        with open(path_to_test / f"{transcription_id}.json", "r") as file:
            transcription = json.load(file)
            transcription_df = pd.DataFrame(transcription, columns=['speaker','text','index'])
            transcription_df['transcription_id'] = transcription_id
        lst_transcription_df.append(transcription_df)
    test_transcriptions_df = pd.concat(lst_transcription_df)
    test_transcriptions_df.to_csv('test_transcriptions_df.csv', index=False)

def create_test_correspondances_df():
    lst_corresondances_df = []
    for transcription_id in test_set:
        with open(path_to_test / f"{transcription_id}.txt", "r") as file:
            correspondance = np.loadtxt(file,delimiter = ' ', dtype = str)
            correspondance_df = pd.DataFrame(correspondance,columns=['1','type','2'])
            correspondance_df['transcription_id'] = transcription_id
        lst_corresondances_df.append(correspondance_df)
    test_correspondances_df = pd.concat(lst_corresondances_df)
    test_correspondances_df.to_csv('test_correspondances_df.csv', index=False)

