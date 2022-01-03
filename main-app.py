# Databricks notebook source
# MAGIC %md
# MAGIC ## Notebook to Invoke the REST API and store the Response body in Parquet Format

# COMMAND ----------

# MAGIC %md
# MAGIC #### Import All Packages

# COMMAND ----------

import json
import os
import sys
import datetime as dt
from dateutil import relativedelta as rt

from requests import Session
from requests.auth import HTTPBasicAuth
from ubw_obfuscation import Obfuscation
import pandas as pd


# COMMAND ----------

# MAGIC %md
# MAGIC #### Setting up Variables

# COMMAND ----------

dbfs = "/dbfs"
mount = "/mnt/data/"
basepath = "dw_data_store/"
obfuscation_rules_file = dbfs + mount + "obfuscation/UBW_API_obfuscation.xlsx"
filename_timestamp = dbutils.widgets.get("filename_timestamp")
year = filename_timestamp[:4]
month = filename_timestamp[4:6]
day = filename_timestamp[6:8]
param_string = dbutils.widgets.get("param_string")
periodic_call_flg = dbutils.widgets.get("periodic_call_flg")
start_period = dbutils.widgets.get("start_period")
end_period = dbutils.widgets.get("end_period")
wild_card_read_flg = dbutils.widgets.get("wild_card_read_flg")
zone = dbutils.widgets.get("zone_name")
source_name = dbutils.widgets.get("source_name")
contry_name = dbutils.widgets.get("country_name")
drop_empty_columns = dbutils.widgets.get("drop_empty_columns")
in_max_timestamp = dbutils.widgets.get("max_timestamp")
in_max_index = dbutils.widgets.get("max_index")
delta_column_name = dbutils.widgets.get("delta_column")
index_column_name = dbutils.widgets.get("index_column")
orig_delta_column_name = dbutils.widgets.get("orig_delta_column")
incremental_mock_flg = dbutils.widgets.get("incremental_mock_flg")
object_name = dbutils.widgets.get("object_name")
transient_object_path = (
    mount
    + basepath
    + zone
    + "/"
    + contry_name
    + "/"
    + source_name
    + "/"
    + object_name
    + "/"
    + year
    + "/"
    + month
    + "/"
    + day
)
jsonl_folderpath = transient_object_path + "/jsonl"
jsonl_filepath = (
    jsonl_folderpath + "/" + object_name + "_" + filename_timestamp + ".jsonl"
)
csv_folderpath = transient_object_path + "/csv"
csv_filepath = csv_folderpath + "/" + object_name + "_" + filename_timestamp + ".csv"
parquet_filepath = (
    transient_object_path + "/" + object_name + "_" + filename_timestamp + ".parquet"
)

print(jsonl_filepath)
# obfuscation_rules_file = 'UBW_API_obfuscation.xlsx'
# jsonl_filepath = 'obfuscated\\' +  object_name + '.jsonl'


# COMMAND ----------

# MAGIC %md
# MAGIC #### Function create_param_query() will create query dict for REST API

# COMMAND ----------

def create_param_query(param_string):
    items = param_string.split("|")
    items_json_list = list()
    for item in items:
        items_json_list.append(json.loads(item))

    params = dict()
    for item in items_json_list:
        if params.get(item["param_key"]) is None:
            params[item["param_key"]] = item["param_value"]
        else:
            params[item["param_key"]] = [params[item["param_key"]]] + [
                item["param_value"]
            ]
    return params


query = create_param_query(param_string)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Setting up variable for any Drop Columns

# COMMAND ----------

drop_column_flg = False
drop_columns_list = list()
if drop_empty_columns == "":
    drop_column_flg = False
else:
    drop_column_flg = True
    drop_columns_list = drop_empty_columns.split(",")
print(drop_columns_list)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Function to create a list of periodic values to invoke multiple calls to REST API

# COMMAND ----------

def create_periodic_list(start_period, end_period, periodic_call_flg):
    period_list = list()
    if periodic_call_flg == "0":
        current_period = year + month
        period_list += [current_period]
        return period_list
    else:
        if end_period == "0":
            current_date = year + month + day
            end_date = dt.datetime.strptime(current_date, "%Y%m%d")
        elif start_period > end_period:
            raise ValueError("Starting period is greater than End period")
        else:
            end_date = dt.date(int(end_period[:4]), int(end_period[4:]), 1)

        start_date = dt.date(int(start_period[:4]), int(start_period[4:]), 1)
        num_months = (end_date.year - start_date.year) * 12 + (
            end_date.month - start_date.month
        )
    print("Number of Months: ", num_months + 1)
    add_month = rt.relativedelta(months=1)
    new_date = start_date
    period_list = [new_date.strftime("%Y%m")]
    for iter in range(num_months):
        new_date = new_date + add_month
        period_list += [new_date.strftime("%Y%m")]

    return period_list


# COMMAND ----------

# MAGIC %md
# MAGIC #### Setting up REST API variables

# COMMAND ----------

url_hostname = dbutils.widgets.get("system_url")
url_pathname = dbutils.widgets.get("url_pathname")
url = url_hostname + url_pathname

header = {
    "content-type": "application/json",
}
print(query)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Function invoke_api to invoke REST request

# COMMAND ----------

def invoke_api(url, header, query):
    session = Session()
    res = session.get(
        url,
        params=query,
        headers=header,
        # timeout=30,
        auth=HTTPBasicAuth(
            dbutils.secrets.get(
                scope="db-kv-scope-lw-caelum-dev-dw", key="UBW-REST-USERID"
            ),
            dbutils.secrets.get(
                scope="db-kv-scope-lw-caelum-dev-dw", key="UBW-REST-PASSWORD"
            ),
        ),
    )
    print("URL Request _:_")
    print(res.url)
    print("Status Code: ", res.status_code)
    if res.status_code != 200:
        return {}
    else:
        return res.json()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Function perform_obfuscation uses Custom Class Obfuscation to Obfuscate the JSON Response body if needed

# COMMAND ----------

def perform_obfuscation(data):
    obfus = Obfuscation()
    return obfus.obfuscated_process(data, object_name, obfuscation_rules_file)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Function create_subfolders() to create the path if not existing

# COMMAND ----------

def create_subfolders(pathname):
    if os.path.exists(pathname):
        print("Path already exists")
    else:
        os.makedirs(pathname)
        print("Path Created ", pathname)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Function write_output_jsonl will write the JSON body into JSONL format

# COMMAND ----------

def write_output_jsonl(data, filepath):
    print("Filepath for JSONL:", filepath)
    with open(filepath, "w", encoding="utf-8") as f:
        for each in data:
            f.write(json.dumps(each) + "\n")
    print("Written the JSONL file in Staging")


# COMMAND ----------

# MAGIC %md
# MAGIC #### Function to persist JSONL and CSV file on ADLS path as Staging data

# COMMAND ----------

def persist_api_response(data, file_qualifier=""):
    create_subfolders(dbfs + transient_object_path + "/jsonl")
    create_subfolders(dbfs + transient_object_path + "/csv")
    qual_jsonl_filepath = jsonl_filepath[:-6] + file_qualifier + jsonl_filepath[-6:]
    write_output_jsonl(data, dbfs + qual_jsonl_filepath)
    pandasDf = pd.DataFrame(data)
    qual_csv_filepath = csv_filepath[:-4] + file_qualifier + csv_filepath[-4:]
    print("Filepath for CSV Data:", qual_csv_filepath)
    pandasDf.to_csv(dbfs + qual_csv_filepath, index=False)
    print("Written data in CSV file")


# COMMAND ----------

# MAGIC %md
# MAGIC #### Main Function to Invoke REST API and persist the JSON response on ADLS

# COMMAND ----------

def invoke_and_persist(url, header, query, file_qualifier=""):
    data = invoke_api(url, header, query)
    len_of_data = len(data)
    print("Number of records in the response body:", len_of_data)
    if len_of_data == 0:
        print("Data can not be written as it is empty response")
    else:
        print("Data will be processed further")
        if object_name == "general-ledger-transactions":  # only for test env
            data = data[:10000]  # only for test env
        # elif object_name == 'suppliers': #only for debugging issue
        # data = data[1:]  #only for debugging issue    data = perform_obfuscation(data)
        persist_api_response(data, file_qualifier)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Run the Main Function to fetch data from REST API

# COMMAND ----------

if object_name == "general-ledger-transactions":
    period_list = create_periodic_list(start_period, end_period, periodic_call_flg)
    for period in period_list:
        filter_param = query["filter"]
        for i, item in enumerate(filter_param):
            if item.startswith("period"):
                query["filter"][i] = "period eq " + period
        file_qualifier = "_" + period
        invoke_and_persist(url, header, query, file_qualifier)
else:
    file_qualifier = ""
    invoke_and_persist(url, header, query, file_qualifier)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Spark Code Starts Here

# COMMAND ----------

# MAGIC %md
# MAGIC #### Importing PySpark Libraries

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import DataFrame, Column
from pyspark.sql.window import Window
from pyspark.sql import functions as sqlfn
from pyspark.sql.functions import (
    explode_outer,
    sha2,
    concat_ws,
    explode,
    size,
    max,
    monotonically_increasing_id,
    lit,
)
from pyspark.sql.types import StructField, StructType, ArrayType, MapType

from typing import List, Dict
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, BooleanType, DateType, ArrayType, IntegerType

import pyspark.sql.functions as F


# COMMAND ----------

# MAGIC %md
# MAGIC #### Modify the CSV and JSONL filepaths to allow multiple files read using spark based on wild cards

# COMMAND ----------

if wild_card_read_flg == "1":
    jsonl_filepath = jsonl_folderpath + "/*"
    csv_filepath = csv_folderpath + "/*"
elif object_name == "general-ledger-transactions":
    jsonl_filepath = jsonl_filepath[:-6] + "*" + jsonl_filepath[-6:]
    csv_filepath = csv_filepath[:-4] + "*" + csv_filepath[-4:]


# COMMAND ----------

# MAGIC %md
# MAGIC #### Define functions for dataframe loading and better schema inference and flattening of struct type

# COMMAND ----------

def flatten_df(schema, prefix=None):
    fields = []
    for field in schema.fields:
        name = prefix + "." + field.name if prefix else field.name
        dtype = field.dataType
        #         if isinstance(dtype, ArrayType):
        #             dtype = dtype.elementType

        if isinstance(dtype, StructType):
            fields += flatten_df(dtype, prefix=name)
        else:
            fields.append(F.col(name).alias(name.replace(".", "_")))

    return fields


def unnest_strunct(dataframe):
    dataframe = dataframe.select(flatten_df(dataframe.schema))
    return dataframe


def load_from_lake(jsonl_filepath, csv_filepath):
    df_json = spark.read.option("inferSchema", "true").json(jsonl_filepath)
    df_csv = spark.read.options(header="True", inferSchema="True", delimiter=",").csv(
        csv_filepath
    )
    types_csv = {f.name: f.dataType for f in df_csv.schema.fields}
    types_json = {f.name: f.dataType for f in df_json.schema.fields}
    for column in types_json:
        if types_csv[column] != types_json[column]:
            if not (
                isinstance(types_json[column], StructType)
                or isinstance(types_json[column], ArrayType)
            ):
                df_json = df_json.withColumn(
                    column, df_json[column].cast(types_csv[column])
                )
    df_json = unnest_strunct(df_json)
    return df_json


# COMMAND ----------

# MAGIC %md
# MAGIC #### Read the CSV and JSONL files on ADLS

# COMMAND ----------

# df = spark.read.options(header='True', inferSchema='True', delimiter=',').csv(csv_filepath)
df = load_from_lake(jsonl_filepath, csv_filepath)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Fundction flattenDataframe to Flatten the StructType datafields only

# COMMAND ----------

def flattenDataframe(df: DataFrame) -> DataFrame:
    fields = df.schema.fields
    fieldNames = list(map(lambda x: x.name, fields))
    for i, field in enumerate(fields):
        fieldtype = field.dataType
        fieldName = field.name
        if isinstance(fieldtype, StructType):
            childFieldnames = list(map(lambda x: fieldName + "." + x, fieldtype.names))
            newfieldNames = (
                list(filter(lambda x: x != fieldName, fieldNames)) + childFieldnames
            )
            renamedcols = list(
                map(
                    lambda x: (sqlfn.col(str(x)).alias(str(x).replace(".", "_"))),
                    newfieldNames,
                )
            )
            explodedf = df.select(renamedcols)
            return flattenDataframe(explodedf)

    return df


# COMMAND ----------

# MAGIC %md
# MAGIC #### Function flattenDataframeWithExplode will flatten Both ArrayType and StructType datatypes resulting in increase number of rows in resulting DataFrame

# COMMAND ----------

def flattenDataframeWithExplode(df: DataFrame) -> DataFrame:
    fields = df.schema.fields
    fieldNames = list(map(lambda x: x.name, fields))
    for i, field in enumerate(fields):
        fieldtype = field.dataType
        fieldName = field.name
        if isinstance(fieldtype, StructType):
            childFieldnames = list(map(lambda x: fieldName + "." + x, fieldtype.names))
            newfieldNames = (
                list(filter(lambda x: x != fieldName, fieldNames)) + childFieldnames
            )
            renamedcols = list(
                map(
                    lambda x: (sqlfn.col(str(x)).alias(str(x).replace(".", "_"))),
                    newfieldNames,
                )
            )
            explodedf = df.select(renamedcols)
            return flattenDataframe(explodedf)
        elif isinstance(fieldtype, ArrayType):
            fieldNamesExcludingArray = list(
                filter(lambda x: x != fieldName, fieldNames)
            )
            fieldNamesAndExplode = fieldNamesExcludingArray + [
                "explode_outer({}) as {}".format(fieldName, fieldName),
            ]
            explodedDf = df.selectExpr(fieldNamesAndExplode)
            return flattenDataframe(explodedDf)

    return df


# COMMAND ----------

# MAGIC %md
# MAGIC #### Acutal Total Count

# COMMAND ----------

actual_count = df.count()
print("Actual Count:", actual_count)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Print Schema for the Resulting DataFrame after Flattening

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Drop the coumns which are empty array based columns

# COMMAND ----------

if drop_column_flg:
    for drop_column in drop_columns_list:
        df = df.drop(drop_column)
df.printSchema()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Converting max_index into integer

# COMMAND ----------

try:
    in_max_index = int(in_max_index)
except Exception as e:
    in_max_index = 0


# COMMAND ----------

# MAGIC %md
# MAGIC #### Filter to check if data is available greater than max_timestamp
# MAGIC #### For periodic_call_flg = '1' , we will not filter and take complete data

# COMMAND ----------

if periodic_call_flg == "1":
    filterDf = df
    filtered_count = actual_count
else:
    filterDf = df.filter(sqlfn.col(orig_delta_column_name) > in_max_timestamp)
    filtered_count = filterDf.count()
print("Filtered Count:", filtered_count)


# COMMAND ----------

# MAGIC %md
# MAGIC #### This is to cater only if incremental mocking is needed
# MAGIC #### Will use the incremental_mock_flg from the metastore
# MAGIC #### Sampling will taken from the whole Dataframe using 5% fraction and seed as current day

# COMMAND ----------

seed_value = dt.datetime.utcnow().day
sample_fraction = 0.05
sampling_flg = False
if incremental_mock_flg == "1" and filtered_count == 0:
    sampleDf = df.sample(fraction=sample_fraction, seed=seed_value)
    sample_count = sampleDf.count()
    print("Sample Count:", sample_count)
    tempDf = sampleDf.withColumn(delta_column_name, sqlfn.current_timestamp())
    sampling_flg = True


# COMMAND ----------

# MAGIC %md
# MAGIC #### Insert ROW_ID in the final Dataframe

# COMMAND ----------

windowSpec = Window.orderBy(delta_column_name)
if not sampling_flg:
    tempDf = filterDf.withColumn(delta_column_name, sqlfn.col(orig_delta_column_name))

newDf = tempDf.withColumn(index_column_name, sqlfn.row_number().over(windowSpec))


# COMMAND ----------

# MAGIC %md
# MAGIC #### Define functions from denormalizing nested records

# COMMAND ----------

def add_hashed_column(dataframe, column_name):
    dataframe = dataframe.withColumn(
        column_name + "_hashed", sha2(dataframe[column_name].cast(StringType()), 512)
    )
    return dataframe


def generate_denormalized_table(dataframe, column_name):
    result = dataframe.select("*", explode_outer(dataframe[column_name])).drop(
        column_name
    )
    result = result.select("*", "col.*").drop("col")
    return result


def denormalizer(dataframe, column_name, identity_columns):
    dataframe = add_hashed_column(dataframe, column_name)
    result = dataframe.select([column_name, column_name + "_hashed", *identity_columns])
    result = generate_denormalized_table(result, column_name)
    return result


# COMMAND ----------

identity_columns = [index_column_name, delta_column_name]
nested_columns = eval(str(dbutils.widgets.get("nested_columns")))
print("Nested Columns:", nested_columns)


# COMMAND ----------

for nested_column in nested_columns:
    max_records_on_array = (
        newDf.select(size(nested_column).alias(nested_column + "len"))
        .groupby()
        .max(nested_column + "len")
        .collect()[0]["max({})".format(nested_column + "len")]
    )
    nestedwindowSpec = Window.partitionBy("ROW_ID").orderBy(
        monotonically_increasing_id()
    )
    nested_parquet_filepath = (
        mount
        + basepath
        + zone
        + "/"
        + contry_name
        + "/"
        + source_name
        + "/"
        + object_name
        + "_"
        + nested_column
        + "/"
        + year
        + "/"
        + month
        + "/"
        + day
        + "/"
        + object_name
        + "_"
        + nested_column
        + "_"
        + filename_timestamp
        + ".parquet"
    )
    if max_records_on_array > 0:
        result = denormalizer(newDf, nested_column, identity_columns)
        result = result.withColumn("ITEM_ID", sqlfn.row_number().over(nestedwindowSpec))
        # added to remove any structypes
        result = flattenDataframe(result)
        result.write.format("parquet").save(nested_parquet_filepath)
        newDf = newDf.withColumn(
            nested_column, sha2(newDf[nested_column].cast(StringType()), 512)
        )
    else:
        print("No nested records found, writing empty file")
        result = newDf.select("contactPoints", *identity_columns).limit(0)

        result = result.withColumn("ITEM_ID", sqlfn.row_number().over(nestedwindowSpec))
        result.write.format("parquet").save(nested_parquet_filepath)
        newDf = newDf.withColumn(nested_column, lit(None).cast(StringType()))


# COMMAND ----------

# MAGIC %md
# MAGIC #### Total Count of the Records

# COMMAND ----------

total_count = newDf.count()
print("Total Count:", total_count)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Fetch Updated values for max_index and max_timestamp

# COMMAND ----------

if total_count > 0:
    out_max_index, out_max_timestamp = newDf.select(
        index_column_name, delta_column_name
    ).rdd.max()
    # Added this since i removed the in_max_timestamp add from above. Therefore out_max_index will reflec only the current
    out_max_index = out_max_index + in_max_index
else:
    out_max_timestamp = in_max_timestamp
    out_max_index = in_max_index
if not isinstance(out_max_timestamp, str):
    out_max_timestamp = out_max_timestamp.isoformat()
print(out_max_timestamp)
print(out_max_index)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Saving the file on the Storage in parquet format

# COMMAND ----------

newDf.coalesce(1).write.parquet(parquet_filepath)
print("Written on Storage:", parquet_filepath)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Creating the Output fields for ADF

# COMMAND ----------

output = {
    "max_index": out_max_index,
    "max_timestamp": out_max_timestamp,
    "record_count": total_count,
}
dbutils.notebook.exit(output)

