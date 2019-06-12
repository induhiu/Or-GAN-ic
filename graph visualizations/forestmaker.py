import pickle

MULT = 1000

d = {1: [(0,0), [2,3], 15 * MULT],
     2: [(4,0), [1,4], 10 * MULT],
     3: [(0,4), [1,4], 20 * MULT],
     4: [(8,4), [2,3], 30 * MULT]}

with open('testdata.txt', 'wb') as fn:
    pickle.dump(d, fn)
