from pickle import load
import numpy as np
import openpyxl as xl

def main():
    ''' The main function '''
    first = np.array(load(open('test1.txt', 'rb')))
    master = []
    for i in range(2, 7):
        master.append(np.array(load(open('test'+str(i)+'.txt', 'rb'))))

    # # --------------------------------------------- #
    # # Analysis
    # # Comparing the diversity of last epoch
    # print(first[-1], second[-1], third[-1], fourth[-3], sep='\n')

    # # Comparing the diversity of last five epochs per test
    # print(first[-1], first[-2], first[-3], first[-4], first[-5], sep='\n')
    # print(second[-1], second[-2], second[-3], second[-4], second[-5], sep='\n')
    # print(third[-1], third[-2], third[-3], third[-4], third[-5], sep='\n')
    # print(fourth[-1], fourth[-2], fourth[-3], fourth[-4], fourth[-5], sep='\n')

    # # Good images for the last one which has noise in it
    # print(fourth[-3], fourth[-6], fourth[-9], fourth[-12], fourth[-15], sep='\n')

    print(master[4][179])

    # # -------------------------------------------------------------- #
    #
    # # Writing them to an excel file for easy visualization
    # alphabet = 'BCDEFGHIJK'
    # wb = xl.Workbook()
    # sheet = wb.active
    # sheet.title = 'Test 1'
    # for i in range(2, 6):
    #     wb.create_sheet(title="Test"+str(i))
    # sheets = wb.get_sheet_names()
    # for i in range(5):
    #     curr_sheet = wb.get_sheet_by_name(sheets[i])
    #     for j in range(2, 182):
    #         curr_sheet['A'+str(j)] = 'Epoch ' + str(j-1)
    #     for j in range(10):
    #         curr_sheet[alphabet[j]+'1'] = alphabet[j]
    #     for j in range(180):
    #         counter = master[i][j][0] if i >= 1 else master[i][j]
    #         for alph in alphabet:
    #             curr_sheet[alph + str(j+2)] = counter[alph]
    #     print("done with sheet, ", curr_sheet)
    # wb.save('results3.xlsx')


if __name__ == '__main__':
    main()
