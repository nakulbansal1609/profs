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
