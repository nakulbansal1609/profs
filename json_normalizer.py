import pandas as pd
import json

#### Read JSON ####
with open("testjson.json", "r") as read_file:
    json_data = json.load(read_file)

#### Normalize JSON ####
def json_normalize(data):
    #### List for ordered column sort ####
    sort_list = list(data[0].keys())
    #### JSON Normalize ####
    df_init = pd.json_normalize(data)
    #### Order Dataframe columns based on innitial JSON keys ####
    init_list = list(df_init.columns.values)
    sorted_list = [y for x in sort_list for y in init_list if x in y]
    return df_init.reindex(columns=sorted_list)

#### Test 
df_test = json_normalize(json_data)