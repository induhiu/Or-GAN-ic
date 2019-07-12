# from ian_gan import GAN, Generator, Discriminator
# import pickle
import openpyxl as xl
import numpy as np

# CITE: https://datascience.stackexchange.com/questions/9262/calculating-kl-divergence-in-python
# DETAILS: Implementation of a KL Score
def KL(a, b):
    a = np.array(a, dtype=np.float)
    b = np.array(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def main():
    ''' The main function '''
    # vals = np.array(pickle.load(open('updated_lang_for_gan.txt', 'rb'))[:60000])
    # vals = np.array(pickle.load(open('lang_for_gan.txt', 'rb'))[:60000])
    # test1_lst = []
    # test1_lst.append(gan1.train(epochs=60))

    # # Test 2: Our method
    # gan1 = GAN()
    # disc, disc2 = Discriminator(), Discriminator()
    # gan2 = GAN(generator=gan1.G, discriminator=disc)
    # gan3 = GAN(generator=gan1.G, discriminator=disc2)
    # test2_lst = []
    # for i in range(20):
    #     test2_lst.append(gan1.train(id=(i*3)+1))
    #     test2_lst.append(gan2.train(id=(i*3)+2))
    #     test2_lst.append(gan3.train(id=(i*3)+3))

    # master = [test1_lst, test2_lst]

    # for i in range(1, 3):
    #     with open('test' + str(i) + '.txt', 'wb') as file:
    #         pickle.dump(master[i - 1], file)

    wb = xl.load_workbook('results5.xlsx')
    sheets = wb.get_sheet_names()
    cols = 'BCDEFGHIJK'
    # a list to hold kl scores
    kl_scores = []
    for i in range(len(sheets)):
        curr_sheet = wb.get_sheet_by_name(sheets[i])
        vals = [(int(curr_sheet[cols[j] + str(183)].value) / 18000) for j in range(10)]
        kl_scores.append(KL(vals, [0.1] * 10))
    print(kl_scores)


main()
