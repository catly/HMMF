

import pandas as pd
import xlrd
import csv

data = xlrd.open_workbook("../data/drug_description.xls")   #打开excel
table1=data.sheets()[0]  #打开excel的第几个sheet
all = []
col1 = table1.col_values(0)
col2 = table1.col_values(1)
for i in range(len(col1)):
    group = []
    group.append(col1[i])
    colq = col2[i].replace('\xa0',' ')
    colq = colq.replace('\xe7',' ')
    colq = colq.replace('\xf6', ' ')
    group.append(colq)
    all.append(group)
with open('../data/drug_description.csv', "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(all)
