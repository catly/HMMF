import numpy as np
import xlrd

data1 = xlrd.open_workbook('../data/drug_side_frequencies.xls')
table1 = data1.sheet_by_name('Sheet1')

data2 = xlrd.open_workbook('../data/drug_id.xls')
table2 = data2.sheet_by_name('Sheet1')

data3 = xlrd.open_workbook('../data/side_id.xls')
table3 = data3.sheet_by_name('Sheet1')

a, b = [], []

d_e = np.zeros((750, 994))

for i in range(table2.nrows):
    x = table2.cell(i, 0).value
    a.append(x)

for i in range(table3.nrows):
    y = table3.cell(i, 0).value
    b.append(y)

for i in range(table1.nrows):
    z1 = table1.cell(i, 0).value
    z2 = table1.cell(i, 1).value
    z3 = table1.cell(i, 2).value
    if z1 in a:
        d_e[a.index(z1), b.index(z2)] = z3



import pickle
lc = '../data/drug_side_frequencies.pkl'
pickle_file = open(lc, 'wb')
pickle.dump(d_e, pickle_file)

pickle_file.close()
