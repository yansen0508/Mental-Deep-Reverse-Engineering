import time
import sys
import os
import glob
import csv
import codecs
import datetime
import random
from psychopy import prefs
prefs.general['audioLib'] = ['pyo']
from psychopy import visual,event,core,gui

import numpy as np
import pickle as Pickle
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sb
from options import Options
from solvers import create_solver

def get_stim_info(file_name, folder):
# read stimulus information stored in same folder as file_name, with a .txt extension
# returns a list of values    
    info_file_name = os.path.join(folder, os.path.splitext(os.path.basename(file_name))[0]+'.txt')
    info = []
    with open(info_file_name,'r') as file:
        reader = csv.reader(file)
        for row in reader:
            info.append(row)
    return info

def enblock(x, n_stims):
    # generator to cut a list of stims into blocks of n_stims
    # returns all complete blocks
    for i in range(len(x)//n_stims):
        start = n_stims*i
        end = n_stims*(i+1)
        yield x[start:end]
    
def generate_trial_files(subject_number,n_blocks,n_stims,iteration):
# generates n_block trial files per subject
# each block contains n_stim trials, randomized from folder which name is inferred from subject_number
# returns an array of n_block file names
    
    seed = time.getTime()
    random.seed(seed+iteration)
    stim_folder = "sounds/Subj"+str(subject_number)
    sound_files = [os.path.basename(x) for x in glob.glob(stim_folder+"/*.jpg")]
    random.shuffle(sound_files)
    first_half = sound_files[:int(len(sound_files)/2)]
    second_half = sound_files[int(len(sound_files)/2):]
    print("sound_files : ",sound_files,len(sound_files))

    block_count = 0
    trial_files = []
#    for block_stims in enblock(list(zip(first_half, second_half)),n_stims):
    for block_stims in enblock(list(zip(first_half, second_half)),n_stims):
        trial_file = 'JJ_trials/non_acc_trials_subj' + str(subject_number) + '_' + str(block_count) + '_' + date.strftime('%y%m%d_%H.%M')+'.csv'
        print ("generate trial file "+trial_file)
        trial_files.append(trial_file)
        with open(trial_file, 'w+') as file :
            # each trial is stored as a row in a csv file, with format: 
            # StimA,MeanA,PA1,PA2,PA3,PA4,PA5,PA6,PA7,StimB,MeanB,PB1,PB2,PB3,PB4,PB5,PB6,PB7
            # where Mean and P1...P7 are CLEESE parameters found in .txt files stored alongside de .wav stims
            # write header
            writer = csv.writer(file)
            writer.writerow(["StimA","StimB"])

            # write each trial in block
            for trial_stims in block_stims:   
                writer.writerow(trial_stims)
        # break when enough blocks
        block_count += 1
        if block_count >= n_blocks:
            break
    return trial_files

def read_trials(trial_file): 
# read all trials in a block of trial, stored as a CSV trial file
    with open(trial_file) as fid :
        print("trial file: ",trial_file)
        reader = csv.reader(fid)
        print("trial header: ",next(reader))
#        for line in reader:
#            print(line)
        trials = list(reader)
        
    return trials[0::] #trim header

def generate_result_file(subject_number):
#AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r, AU10_r, AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r, AU25_r, AU26_r
    result_file = os.path.dirname(__file__)+'/JJ_results/non_acc_results_subj'+str(subject_number)+'_'+date.strftime('%y%m%d_%H.%M')+'.csv'        
    result_headers = ['subj','trial','block', 'sex', 'age', 'date', 'image','AUs (AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r, AU10_r, AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r, AU25_r, AU26_r)',
                      'decision','reponse time']
    with open(result_file, 'w+') as file:
        writer = csv.writer(file)
        writer.writerow(result_headers)
    return result_file

def show_text_and_wait(file_name = None, message = None):
    event.clearEvents()
    if message is None:
        with codecs.open (file_name, "r", "utf-8") as file :
            message = file.read()
    text_object = visual.TextStim(win, text = message, color = 'black')
    text_object.height = 0.05
    text_object.draw()
    win.flip()
    while True :
        if len(event.getKeys()) > 0: 
            core.wait(0.2)
            break
        event.clearEvents()
        core.wait(0.2)
        text_object.draw()
        win.flip()

def show_text(file_name = None, message = None):
    if message is None:
        with codecs.open (file_name, "r", "utf-8") as file :
            message = file.read()
    text_object = visual.TextStim(win, text = message, color = 'black')
    text_object.height = 0.05
    text_object.draw()
    win.flip()

def update_trial_gui(): 
    play_instruction.draw()
    play_icon.draw()
    play_icon1.draw()
    response_instruction.draw()

    for response_label in response_labels: response_label.draw()
    for response_checkbox in response_checkboxes: response_checkbox.draw()
    win.flip()
 
    
au2ix={'1':0, '2':1, '4':2, '5':3, '6':4, '7':5,
     '9':6, '10':7, '12':8, '14':9, '15':10, '17':11, 
     '20':12, '23':13, '25':14, '26':15,'happy':16}

n_blocks = 1
n_stims = 280
au_label = Pickle.load(open('aus_openface_new560.pkl','rb'),encoding='iso-8859-1')
img_path = os.path.dirname(__file__)+'/images/'
sound_path = os.path.dirname(__file__)+'/sounds/'
#print("path: ",img_path)
# get participant nr, age, sex 
subject_info = {u'Number':1, u'Age':20, u'Sex': u'f/m'}
dlg = gui.DlgFromDict(subject_info, title=u'REVCOR')
if dlg.OK:
    subject_number = subject_info[u'Number']
    subject_age = subject_info[u'Age']
    subject_sex = subject_info[u'Sex']    
else:
    core.quit() #the user hit cancel so exit
date = datetime.datetime.now()
time = core.Clock()

# create stimuli if folder don't exist
# warning: if folder exists with wrong number of stims

output_folder = sound_path + 'Subj' + str(subject_number)

#if not os.path.exists(output_folder):
#	generate_stimuli(subject_number, n_blocks=n_blocks, n_stims=n_stims, base_sound='./sounds/male_vraiment_flat.wav', config_file='./config.py')

win = visual.Window([1366,768],fullscr=False,color="lightgray", units='norm')
screen_ratio = (float(win.size[1])/float(win.size[0]))
isi = .5

# trial gui
question = u'Which face is more self-confident?'
response_options = ['[g] Left','[h] Right']
response_keys = ['g', 'h']
label_size = 0.07
play_instruction = visual.TextStim(win, units='norm', text='Here are two faces', color='red', height=label_size, pos=(0,0.5))
response_instruction = visual.TextStim(win, units='norm', text=question, color='black', height=label_size, pos=(0,0.1), alignHoriz='center')
play_icon = visual.ImageStim(win, image=img_path+'play_on.png', units='norm', size = (1*screen_ratio,1), pos=(-0.5,0.5+0.25*label_size))
play_icon1 = visual.ImageStim(win, image=img_path+'play_off.png', units='norm', size = (1*screen_ratio,1), pos=(0.5,0.5+0.25*label_size))
response_labels = []
response_checkboxes = []
reponse_ypos = -0.2
reponse_xpos = -0.1
label_spacing = abs(-0.8 - reponse_ypos)/(len(response_options)+1)
for index, response_option in enumerate(response_options):
    y = reponse_ypos - label_spacing * index
    response_labels.append(visual.TextStim(win, units = 'norm', text=response_option, alignHoriz='left', height=label_size, color='black', pos=(reponse_xpos,y)))
    response_checkboxes.append(visual.ImageStim(win, image=img_path+'rb_off.png', size=(label_size*screen_ratio,label_size), units='norm', pos=(reponse_xpos-label_size, y-label_size*.05)))

# generate data files
result_file = generate_result_file(subject_number)

# experiment 
show_text_and_wait(file_name="intro.txt")  
show_text_and_wait(file_name="practice.txt")  
trial_count = 0

intern_count = 0

first_idx = 0
second_idx = 1
iteration = 3
all_trial=iteration*n_stims

while (iteration > 0) :
    trial_files = generate_trial_files(subject_number,n_blocks,n_stims,iteration)
    for block_count, trial_file in enumerate(trial_files):
        block_trials = read_trials(trial_file)
        for trial in block_trials :
                
            play_instruction.setColor('black')
            
            for checkbox in response_checkboxes:
                checkbox.setImage(img_path+'rb_off.png')
            sound_1 = output_folder +'/'+trial[0]
            sound_2 = sound_path+'Subj'+str(subject_number)+'/'+trial[1]
            end_trial = False
            while (not end_trial):

                play_icon.setImage(sound_1)
                play_icon1.setImage(sound_2)
                # focus response instruction
                response_start = time.getTime()
                response_instruction.setColor('red')
                update_trial_gui()
                # upon key response...
                response_key = event.waitKeys(keyList=response_keys)
                response_time = time.getTime() - response_start
                # unfocus response_instruction, select checkbox
                response_instruction.setColor('black')
                response_checkboxes[response_keys.index(response_key[0])].setImage(img_path+'rb_on.png')
                update_trial_gui()
                # blank screen and end trial
                core.wait(0.2) 
                win.flip()
                core.wait(0.1) 
                end_trial = True
            
            row = [subject_number, trial_count, block_count, subject_sex, subject_age, date]
            if response_key == ['g']:
                response_choice = 0
            elif response_key == ['h']:
                response_choice = 1
            # write down the results        
            with open(result_file, 'a') as file :
                writer = csv.writer(file,lineterminator='\n')

                for stim_order,stim in enumerate(trial):
                    print("stim_order , stim ",stim_order,stim)
                    result = row + [stim,au_label[stim[2:-4]],response_choice==stim_order,round(response_time,3)]
                    writer.writerow(result)

            intern_count += 1        
            trial_count += 1   
            print("response done, remain ",all_trial - trial_count," trials")

         
        # inform end of practice at the end of first block
        if block_count == 0:
           
           show_text_and_wait("pause1.txt")
           show_text_and_wait("pause0.txt")   
    
    iteration -=1

show_text_and_wait(file_name="end_practice.txt")         
   
#End of experiment
show_text_and_wait("end.txt")

# Close Python
win.close()
core.quit()
sys.exit()
