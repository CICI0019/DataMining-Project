import pandas as pd
import numpy as np

data_test = pd.read_excel('data/adulttest.xlsx')
data_input = pd.read_excel('data/adultinput.xlsx')

data_test #adulttest.xlsx TEST DATA
data_input #adultinput.xlsx TRAINING DATA


# 1) PRE-PROCESSING FOR DATA_INPUt

# 1. Remove Rows with Missing Values
replaced = data_input.replace('?', np.nan) # replaces '?' with 'NaN'
cleanDataInput = replaced.dropna() # drops values 'NaN'

# 2. Remove attributes 'fnlwgt', 'education', and 'relationship'
cleanDataInput.drop(['fnlwgt', 'education', 'relationship'], axis=1, inplace = True)
df_input = cleanDataInput

# 3.1 Binarize the 'capital_gain' & 'capital_loss' attributes

#binarization of capital gain, capital loss, and native country attributes

df_input["capital_gain"] = np.where(df_input["capital_gain"] > 0, 1,0)
df_input["capital_loss"] = np.where(df_input["capital_loss"] > 0, 1,0)

#Binarization of sex attribute
df_input["sex=female"] = np.where(df_input["sex "] == "Female", 1, 0)
df_input["sex=male"] = np.where(df_input["sex "] == "Male", 1, 0)

df_input.drop("sex ", axis=1, inplace=True)

# Asymmetric Binarization of race attribute
df_input["race=white"] = np.where(df_input["race"] == "White", 1, 0)
df_input["race=asian-pac-islander"] = np.where(df_input["race"] == "Asian-Pac-Islander", 1, 0)
df_input["race=black"] = np.where(df_input["race"] == "Black", 1, 0)
df_input["race=amer-indian-eskimo"] = np.where(df_input["race"] == "Amer-Indian-Eskimo", 1, 0)
df_input["race=other"] = np.where(df_input["race"] == "Other", 1, 0)

df_input.drop("race", axis=1, inplace=True)

# Binarization of country attribute
df_input["country"] = np.where(df_input["country"] == "United-States", 1, 0)


# 3.2 Discretize Continuous Attributes of Age Column

df_input["young"] = np.where(df_input["age"] <= 25, 1,0)
df_input["adult"] = np.where((df_input["age"] >= 26) & (df_input["age"] <= 45), 1,0)
df_input["senior"] = np.where((df_input["age"] >= 46) & (df_input["age"] <= 65), 1,0)
df_input["old"] = np.where((df_input["age"] >= 66) & (df_input["age"] <= 90), 1,0)

df_input.drop("age", axis=1, inplace=True)

# 3.2 Discretize Continuous Attributes of Hr_Per_Week Column

df_input["part_time"] = np.where(df_input["hr_per_week"] < 40, 1, 0)
df_input["full_time"] = np.where(df_input["hr_per_week"] == 40, 1, 0)
df_input["overtime"] = np.where(df_input["hr_per_week"] > 40, 1, 0)

df_input.drop("hr_per_week", axis=1, inplace=True)

# 3.3 Merge type_employer column values

df_input["type_employer "] = df_input["type_employer "].replace(to_replace=['Local-gov','State-gov', 'Federal-gov'], value='gov')
df_input["type_employer "] = df_input["type_employer "].replace(to_replace=['Without-pay', 'Never-worked'], value='Not-Working')
df_input["type_employer "] = df_input["type_employer "].replace(to_replace=['Self-emp-not-inc', 'Self-emp-inc'], value='Self Employeed')

# Binarization
df_input["gov"] = np.where(df_input["type_employer "] == 'gov', 1, 0)
df_input["Not-Working"] = np.where(df_input["type_employer "] == 'Not-Working', 1, 0)
df_input["Private"] = np.where(df_input["type_employer "] == 'Private', 1, 0)
df_input["Self Employed"] = np.where(df_input["type_employer "] == 'Self Employed', 1, 0)

df_input.drop("type_employer ", axis=1, inplace=True)

# 3.3 Merge Marital column values

df_input["marital "] = df_input["marital "].replace(to_replace=['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'], value='Married')
df_input["marital "] = df_input["marital "].replace(to_replace=['Never-married', 'Divorced', 'Separated', 'Widowed'], value='Not Married')

# Binarization
df_input["Married"] = np.where(df_input["marital "] == 'Married', 1, 0)
df_input["Never-married"] = np.where(df_input["marital "] == 'Never-married', 1, 0)
df_input["Not Married"] = np.where(df_input["marital "] == 'Not Married', 1, 0)

df_input.drop("marital ", axis=1, inplace=True)

# 3.3 Merge Occupation Column Values

df_input['occupation'] = df_input['occupation'].replace(to_replace=['Tech-support', 'Adm-clerical', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', 'Other-service'], value='Other')
df_input['occupation'] = df_input['occupation'].replace(to_replace=['Craft-repair', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Transport-moving'], value='ManualWork')

# Binarization
df_input["Exec-managerial"] = np.where(df_input["occupation"] == 'Exec-managerial', 1, 0)
df_input["Prof-specialty"] = np.where(df_input["occupation"] == 'Prof-specialty', 1, 0)
df_input["Other"] = np.where(df_input["occupation"] == 'Other', 1, 0)
df_input["ManualWork"] = np.where(df_input["occupation"] == 'ManualWork', 1, 0)
df_input["Sales"] = np.where(df_input["occupation"] == 'Sales', 1, 0)

df_input.drop("occupation", axis=1, inplace=True)

# 1) PRE-PROCESSING FOR DATA_TEST

income_column = data_test["income"]
data_test.drop("income", axis=1, inplace=True)

# 1. Remove Rows with Missing Values
replaced_test = data_test.replace('?', np.nan) # replaces '?' with 'NaN'
cleanDataTest = replaced_test.dropna() # drops values 'NaN'

# 2. Remove attributes 'fnlwgt', 'education', and 'relationship'
cleanDataTest.drop(['fnlwgt', 'education', 'relationship'], axis=1, inplace = True)
df_test = cleanDataTest

# 3.1 Binarize the 'capital_gain' & 'capital_loss' attributes

#binarization of capital gain, capital loss, and native country attributes

df_test["capital_gain"] = np.where(df_test["capital_gain"] > 0, 1,0)
df_test["capital_loss"] = np.where(df_test["capital_loss"] > 0, 1,0)

#Binarization of sex attribute
df_test["sex=female"] = np.where(df_test["sex "] == "Female", 1, 0)
df_test["sex=male"] = np.where(df_test["sex "] == "Male", 1, 0)

df_test.drop("sex ", axis=1, inplace=True)

# Asymmetric Binarization of race attribute
df_test["race=white"] = np.where(df_test["race"] == "White", 1, 0)
df_test["race=asian-pac-islander"] = np.where(df_test["race"] == "Asian-Pac-Islander", 1, 0)
df_test["race=black"] = np.where(df_test["race"] == "Black", 1, 0)
df_test["race=amer-indian-eskimo"] = np.where(df_test["race"] == "Amer-Indian-Eskimo", 1, 0)
df_test["race=other"] = np.where(df_test["race"] == "Other", 1, 0)

df_test.drop("race", axis=1, inplace=True)

# Binarization of country attribute
df_test["country"] = np.where(df_test["country"] == "United-States", 1, 0)


# 3.2 Discretize Continuous Attributes of Age Column

df_test["young"] = np.where(df_test["age"] <= 25, 1,0)
df_test["adult"] = np.where((df_test["age"] >= 26) & (df_test["age"] <= 45), 1,0)
df_test["senior"] = np.where((df_test["age"] >= 46) & (df_test["age"] <= 65), 1,0)
df_test["old"] = np.where((df_test["age"] >= 66) & (df_test["age"] <= 90), 1,0)

df_test.drop("age", axis=1, inplace=True)

# 3.2 Discretize Continuous Attributes of Hr_Per_Week Column

df_test["part_time"] = np.where(df_test["hr_per_week"] < 40, 1, 0)
df_test["full_time"] = np.where(df_test["hr_per_week"] == 40, 1, 0)
df_test["overtime"] = np.where(df_test["hr_per_week"] > 40, 1, 0)

df_test.drop("hr_per_week", axis=1, inplace=True)

# 3.3 Merge type_employer column values

df_test["type_employer "] = df_test["type_employer "].replace(to_replace=['Local-gov','State-gov', 'Federal-gov'], value='gov')
df_test["type_employer "] = df_test["type_employer "].replace(to_replace=['Without-pay', 'Never-worked'], value='Not-Working')
df_test["type_employer "] = df_test["type_employer "].replace(to_replace=['Self-emp-not-inc', 'Self-emp-inc'], value='Self Employeed')

# Binarization
df_test["gov"] = np.where(df_test["type_employer "] == 'gov', 1, 0)
df_test["Not-Working"] = np.where(df_test["type_employer "] == 'Not-Working', 1, 0)
df_test["Private"] = np.where(df_test["type_employer "] == 'Private', 1, 0)
df_test["Self Employed"] = np.where(df_test["type_employer "] == 'Self Employed', 1, 0)

df_test.drop("type_employer ", axis=1, inplace=True)

# 3.3 Merge Marital column values

df_test["marital "] = df_test["marital "].replace(to_replace=['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'], value='Married')
df_test["marital "] = df_test["marital "].replace(to_replace=['Never-married', 'Divorced', 'Separated', 'Widowed'], value='Not Married')

# Binarization
df_test["Married"] = np.where(df_test["marital "] == 'Married', 1, 0)
df_test["Never-married"] = np.where(df_test["marital "] == 'Never-married', 1, 0)
df_test["Not Married"] = np.where(df_test["marital "] == 'Not Married', 1, 0)

df_test.drop("marital ", axis=1, inplace=True)

# 3.3 Merge Occupation Column Values

df_test['occupation'] = df_test['occupation'].replace(to_replace=['Tech-support', 'Adm-clerical', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', 'Other-service'], value='Other')
df_test['occupation'] = df_test['occupation'].replace(to_replace=['Craft-repair', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Transport-moving'], value='ManualWork')

# Binarization
df_test["Exec-managerial"] = np.where(df_test["occupation"] == 'Exec-managerial', 1, 0)
df_test["Prof-specialty"] = np.where(df_test["occupation"] == 'Prof-specialty', 1, 0)
df_test["Other"] = np.where(df_test["occupation"] == 'Other', 1, 0)
df_test["ManualWork"] = np.where(df_test["occupation"] == 'ManualWork', 1, 0)
df_test["Sales"] = np.where(df_test["occupation"] == 'Sales', 1, 0)

df_test.drop("occupation", axis=1, inplace=True)

df_test["income"] = '?'

# SHOWS THE SIZE AND DIMENSIONALITY OF THE DATA SETS
print("\nSize and Dimensionality of the Data Sets:\n")
print("Input Data:", df_input.shape)
print("Test Data:", df_test.shape,"\n")