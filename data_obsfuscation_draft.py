import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from random import randrange, uniform
import re

################### Numeric Obfuscation ###############
#### Int Value ####
irand = randrange(0, 10)
#### Float Value ####
frand = uniform(0, 10)

##### Functions to be called in Data Obfuscation ####
##### Datetime Obfuscation #####
def add_days(curr_date,variance):
    curr_date = curr_date + timedelta(variance)
    return curr_date
def subtract_days(curr_date,variance):
    curr_date = curr_date - timedelta(variance)
    return 
##### Numeric Obfuscation #####
def add_number(curr_number,variance):
    curr_number = curr_number + variance
    return curr_number
def subtract_number(curr_number,variance):
    curr_number = curr_number - variance
    return curr_number
##### String Obfuscation #####
def num_obs(lst):
    if lst.isdigit():
        lst_fin = ''.join([random.choice(string.digits) for ind in range(len(lst))])
    elif lst.isalpha():
        lst_fin = ''.join([random.choice(ascii_letters) for ind in range(len(lst))])
    else:
        lst_fin = lst
    return lst_fin
def string_obfuscation(value,filter):
    list_substrings = re.findall(r'[A-Za-z]+|\d+| |[^\w\s]', value)
    if filter=='DEFAULT':
        list_substrings = [num_obs(x) for x in list_substrings]
        obfus_string = ''.join(list_substrings)
    elif filter == 'only_char':
        list_substrings = [num_obs(x) if x.isalpha() else x for x in list_substrings]
        obfus_string = ''.join(list_substrings)
    elif filter == 'only_num':
        list_substrings = [num_obs(x) if x.isdigit() else x for x in list_substrings]
        obfus_string = ''.join(list_substrings)
    else:
        obfus_string = None
    return obfus_string

#### Data Obfuscation ####
def data_obsfuscation(value):

    #### String Obfuscation ####
    value = string_obfuscation(value,'DEFAULT')
    #### Numeric Obfuscation ####
    if type(value) is np.int64:
        value = random.choice([add_number(x,randrange(0, x)),subtract_number(x,randrange(0, x))])
    #### DatetimeObfuscation ####
    elif type(value) is pd.Timestamp:
        value = random.choice([add_days(x,random.randint(1,30)),subtract_days(x,random.randint(1,30))])
    #### RandomItem ####
    elif type(value) is list:
        value = random.choice(value)
    #### RandomBool ####
    elif type(value) is None:
        value = random.randint(0, 1)
    else:
        value = None
    #### Return value #####
    return value
