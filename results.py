from pickle import load
import numpy as np
import openpyxl as xl

def main():
    ''' The main function '''
    # first = np.array(load(open('test1.txt', 'rb')))
    master = []
    for i in range(3, 8):
        master.append(np.array(load(open('test'+str(i)+'.txt', 'rb'))))
    # for ary in master:
    #     print(ary.shape)
    # print(master[3][-5:])
    # for _ in range(5):
    #   print()
    # print(master[4][-5:])

    
    # Writing them to an excel file for easy visualization
    # alphabet = 'BCDEFGHIJK'
    wb = xl.Workbook()
    sheet = wb.active
    sheet.title = 'Test 3'
    for i in range(4, 8):
        wb.create_sheet(title="Test"+str(i))
    sheets = wb.get_sheet_names()
    for i in range(5):
        alphabet = 'BCDEFGHIJK' if i <= 2 else 'ABCDEFGHIJ'
        curr_sheet = wb.get_sheet_by_name(sheets[i])
        for j in range(2, 182):
            curr_sheet['A'+str(j)] = 'Epoch ' + str(j-1)
        for j in range(10):
            curr_sheet['BCDEFGHIJK'[j]+'1'] = alphabet[j]
        for j in range(180):
            counter = master[i][j][0] if i != 3 else master[i][0][j]
            # counter = master[i][j][0]
            for x in range(len(alphabet)):
                key, val = 'BCDEFGHIJK'[x] + str(j+2), counter[alphabet[x]]
                curr_sheet[key] = val
        print("done with sheet, ", curr_sheet)
    wb.save('results5.xlsx')


if __name__ == '__main__':
    main()
`