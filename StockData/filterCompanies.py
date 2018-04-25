import pandas as pd
import numpy as np
import csv
import glob, os
from itertools import groupby
import random


#filtering out required columns 
df=pd.read_csv("rawData.csv")
keep_col = ['datadate','conm','prccd']
new_df = df[keep_col]
new_df.to_csv("stockDataFiltered.csv", index=False)


#Adding two extra rows after each firm for calculation correctness
firmx = []
with open('stockDataFiltered.csv') as f:
    reader = csv.reader(f)
    with open('stockDataMergedTemp.csv', 'w') as g:
        writer = csv.writer(g)
        next(reader, None)
        writer.writerow(['date', 'firm', 'closingPrice'])
        for row in reader:
        	if row[1] not in firmx:
        		firmx.append(row[1])
        		new_row =  [ row[1]]
        		writer.writerow(new_row)
        		writer.writerow(new_row)
        	writer.writerow(row)


#diff calculation
df = pd.read_csv('stockDataMergedTemp.csv')
#df.drop(df.head(2).index, inplace=True)
df['diff'] = df['closingPrice'].shift(-2) - df['closingPrice'].shift(2)
df.to_csv('stockDataReorderedWithDiff.csv', index=False, sep=',')

#drop empty columns with empty cells 
df = pd.read_csv('stockDataReorderedWithDiff.csv')
df.dropna().to_csv('stockDataReorderedFinal.csv', index=False, sep=',')


#Dropping columns with 0 diff 
with open('stockDataReorderedFinal.csv') as inp, open('stockDataFinalEnhanced.csv', 'w') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if (row[3] != "0.0"):
            writer.writerow(row)



#Filtering out companies in the 4 years 
firm = []
year = []
count = 0;
with open('stockDataFinalEnhanced.csv') as f:
    reader = csv.reader(f)
    with open('FinalCompanies.csv', 'w') as g:
        writer = csv.writer(g)
        next(reader, None)
        for row in reader:
            if row[1] not in firm:
                count = count + 1 
                y = (row[0].split("/"))[2]
                if y not in year:
                    year.append(y)
                if len(y) == 4:
                    firm.append(row[1])
                    new_row =  [row[1]]
                    writer.writerow(new_row)
        print("Total firms in data: ",count)




#Randomly choosing 500 companies' data depending on the above filtered companies 
random.shuffle(firm)
firm = firm[0:501]
with open('stockDataFinalEnhanced.csv') as f:
    reader = csv.reader(f)
    with open('stockDataFinalEnhanced500.csv', 'w') as g:
        writer = csv.writer(g)
        next(reader, None)
        for row in reader:
            if row[1] in firm:
                writer.writerow(row)


"""
#TODO : Debug | Code for increasing desirable rows
df = pd.read_csv('stockDataReorderedWithDiff.csv')
df['diff2'] = np.where(df['diff'].isnull, df['closingPrice'].shift(-1) - df['closingPrice'].shift(1), " ")
df.to_csv('stockDataReorderedWithDiff2.csv', index=False, sep=',')

"""










