import os
import sys
import pandas as pd 
import json
from yaml import safe_load
import openpyxl
import re
from flatten_json import flatten, unflatten_list

import random


#CONFIG_FILE = 'config.yaml'
CONFIG_FILE = 'configtest.yaml'
WORKBOOK = 'UBW_API_obfuscation.xlsx'
CWD = os.getcwd()

with open(CONFIG_FILE, 'r') as f:
    config = safe_load(f)

INPUT_PATH = os.path.join(CWD, config['input-path'])
OUTPUT_PATH = os.path.join(CWD, config['output-path'])
OBFUSCATION_LOGS = os.path.join(CWD, config['obfuscation-logs'])
NESTING_SEP = config['nesting-separator']
#print(config)
wb = openpyxl.load_workbook('UBW_API_obfuscation.xlsx')
ws = wb.active
sheet = ws.values
cols = next(sheet)
sheet = list(sheet)
sheetDF = pd.DataFrame(sheet, columns=cols)

def obfuscated_process(entity_name, filename):
    print('Reading file: ', filename)
    actual_list_of_dict = read_json_file(filename)

    print('Flattening Data... ')
    flattened_list_of_dict = generate_flatten_list(actual_list_of_dict)
    obfuscated_flt_list_of_dict = flattened_list_of_dict.copy()
    
    nested_array_cols_dict = generate_flt_col_names(flattened_list_of_dict)

    entitySheetDF = sheetDF[sheetDF['Entity'] == entity_name]
    for index, each_dict in enumerate(flattened_list_of_dict):
        for each_key in each_dict.keys():
            nested_nonarray_col = nested_array_cols_dict.get(each_key)
            if nested_nonarray_col is None:
                current_key = each_key
            else:
                current_key = nested_nonarray_col
            
            obfuscation_type = \
                entitySheetDF['Obfuscation applied'][entitySheetDF['Column'] == current_key][:1]
            if len(obfuscation_type) == 0:
                pass #todo
            actual_value = each_dict[each_key]
            if actual_value is not None:
                obfuscated_value = perform_obfuscation(actual_value, obfuscation_type)
                if obfuscated_value is None:
                    print("Obfuscation not done..")
                else:
                    obfuscated_flt_list_of_dict[index][each_key] = \
                        obfuscated_value




    print('Unflattening Data...')
    unflattened_list_of_dict = generate_unflatten_list(obfuscated_flt_list_of_dict)
    
    print('Writing Data: ', filename)
    write_json_file(filename, unflattened_list_of_dict)


"""
This function converts the list of json/dicts into a list of json/dict
with the flattened structures
"""
def generate_flatten_list(actual_list_of_dict):
    return [flatten(each_dict, NESTING_SEP) for each_dict in actual_list_of_dict]

"""
This function identifies those keys in a dictionary which have the keys
like a pattern '.\d{1}.' . This is needed to know which keys were having 
the nested array structure before flattening
"""
def generate_flt_col_names(flattened_list_of_dict):
    dataDF = pd.DataFrame(flattened_list_of_dict)
    nested_array_cols_dict = dict()
    col_names = list(dataDF.columns)
    pattern = r'_\d{1}_'
    for each_col in col_names:
        if re.search(pattern, each_col) is not None:
            value = re.sub(pattern, '_', each_col)
            nested_array_cols_dict[each_col] = value
            
    return nested_array_cols_dict


"""
The function will perform the actual obfuscation on the python
object/string passsed inside the function
"""
def perform_obfuscation(actual_value, obfuscation_type):
    if len(obfuscation_type) == 0:
        obfuscated_value = actual_value
    else:
        obfuscation_rule = list(obfuscation_type)[0]
        obfuscated_value = data_obsfuscation(actual_value, obfuscation_rule)
        print(actual_value, obfuscated_value)

    return obfuscated_value



"""
This function unflattens the already flattened llist of json/dict into the 
list of json/dict with their previous nested structuring.
"""
def generate_unflatten_list(flattened_list_of_dict):
    return [unflatten_list(each_dict, NESTING_SEP) for each_dict in flattened_list_of_dict]


"""
The function will read the json file with the input as a filename
and return the data back with python based object list/dict
"""
def read_json_file(filename):
    filepath = os.path.join(INPUT_PATH, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


"""
The function will write the python object list/dict into
the json format output file
"""
def write_json_file(filename, json_data):
    filepath = os.path.join(OUTPUT_PATH, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(json_data))




##########################
##### Functions to be called in Data Obfuscation ####
##### Rule Processing #####
def rule_procc(rule):
    import re
    ####
    if rule.startswith('String'):
        result = re.search(r'\(([^()]*)\)$', rule).group(1).split(',')[1].strip().replace("'", '')
        rule = 'string'
        return (rule,result)
    elif rule.startswith('Numeric'):
        result = re.search(r'\(([^()]*)\)$', rule).group(1).split(',')[1].strip().replace("'", '')
        rule='numeric'
        return (rule,result)
    elif rule.startswith('Datetime'):
        result = re.search(r'\(([^()]*)\)$', rule).group(1).split(',')[1].strip().replace("'", '')
        rule='datetime'
        return (rule,result)
    elif rule.startswith('DateOb'):
        result = re.search(r'\(([^()]*)\)$', rule).group(1).split(',')[1].strip().replace("'", '')
        rule='date'
        return (rule,result)
    elif rule.startswith('DateYY'):
        result = re.search(r'\(([^()]*)\)$', rule).group(1).split(',')[1].strip().replace("'", '')
        rule='date'
        return (rule,result)
    elif rule.startswith('RandomItem'):
        rule='randomItem'
        result = ''
        return (rule,result)
    elif rule.startswith('RandomBool'):
        rule = 'randomBool'
        result = ''
        return (rule,result)
    else:
        raise ValueError("Rule passed not in the list of defined ones")
##### Datetime Obfuscation #####
def add_days(curr_date,variance,rule):
    from datetime import datetime, timedelta
    ####
    if type(curr_date) is str:
        if rule == 'datetime':
            curr_date = datetime.strptime(curr_date, '%Y-%m-%d %H:%M:%S.%f') + timedelta(int(variance))
        elif rule == 'date':
            if curr_date == '0':
                return curr_date
            else:
                curr_date = datetime.strftime(datetime.strptime(curr_date, '%Y%m') + timedelta(int(variance)), '%Y%m')
    return curr_date
def subtract_days(curr_date,variance,rule):
    from datetime import datetime, timedelta
    ####
    if type(curr_date) is str:
        if rule == 'datetime':
            curr_date = datetime.strptime(curr_date, '%Y-%m-%d %H:%M:%S.%f') - timedelta(int(variance))
        elif rule == 'date':
            if curr_date == '0':
                return curr_date
            else:
                curr_date = datetime.strftime(datetime.strptime(curr_date, '%Y%m') + timedelta(int(variance)), '%Y%m')
    return curr_date
##### Numeric Obfuscation #####
def add_number(curr_number,variance):
    if type(curr_number) is str:
        curr_number = int(curr_number)
        curr_number += int(variance)
    elif (type(curr_number) is float) or (type(curr_number) is int):
        curr_number += int(variance)
    return curr_number
def subtract_number(curr_number,variance):
    if type(curr_number) is str:
        curr_number = int(curr_number)
        curr_number -= int(variance)
    elif (type(curr_number) is float) or (type(curr_number) is int):
        curr_number -= int(variance)
    return curr_number
##### String Obfuscation #####
def num_obs(lst):
    import string
    ####
    if lst.isdigit():
        lst_fin = ''.join([random.choice(string.digits) for ind in range(len(lst))])
    elif lst.isalpha():
        lst_fin = ''.join([random.choice(string.ascii_letters) for ind in range(len(lst))])
    else:
        lst_fin = lst
    return lst_fin
def string_obfuscation(value,filter):
    import re
    ####
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
def data_obsfuscation(value, rule):
    import string
    import re
    from datetime import datetime, timedelta
    import random
    #### Process Rule ####
    rule,result = rule_procc(rule)
    #### String Obfuscation ####
    if rule == 'string':
        value = string_obfuscation(value,result)
    #### Numeric Obfuscation ####
    elif rule == 'numeric':
        value = random.choice([add_number(value,result),subtract_number(value,result)])
    #### DateObfuscation && DatetimeObfuscation ####
    elif (rule == 'date') or (rule == 'datetime'):
        value = random.choice([add_days(value,result,rule),subtract_days(value,result,rule)])
    #### RandomItem ####
    elif rule == 'randomItem':
        value = random.choice(value)
    #### RandomBool ####
    elif rule == 'randomBool':
        value = random.randint(0, 1)
    else:
        value = None
    #### Return value #####
    return value


#############################


"""
This is hte main program call
"""

def main():
    if len(sys.argv) > 1:
        entity_arg = sys.argv[1]
        if config['entity'].get(entity_arg) is None:
            print('Entity not found in config file')
            exit(1)
    else:
        entity_arg = None
    
    if entity_arg is None:
        for each_entity, file_list in config['entity'].items():
            for each_file in file_list:
                obfuscated_process(each_entity, each_file)
    else:
        file_list = config['entity'].get(entity_arg)
        for each_file in file_list:
            obfuscated_process(entity_arg, each_file)



if __name__ == '__main__':
    main()


