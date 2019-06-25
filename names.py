from pickle import dump

names = ['The Great Deku Tree', "Treebeard", "The Giving Tree", "Groot", "Kite-Eating Tree",
         'Whomping Willow', "White Tree of Gondor", "The Tree's Knees", "Grandmother Willow",
         "The Man From Another Place", "Mokujin", "Old Man Willow", "Truffula Tree",
         "The Tumtum Tree", 'Everett', 'Silas', 'Oliver', 'Alfred', 'Lillian', 'Bernard',
         'Archie', 'Bernadette', 'Clarence', 'Alice', 'Claude', 'Olive', 'Edgar', 'Rose',
         'Augustine', 'Nora', 'Chester', 'Marjorie', 'Grover', 'Charlotte', 'Blanche',
         'Edith', 'Harriet', 'Josephine', 'Mabel', 'Minnie', 'Pearl', 'Ruth', 'Sadie',
         'Violet', 'Millie', 'Beatrice', 'Vera']
for l in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    for i in range(1, 11):
        names.append(l * i)

with open('names.txt', 'wb') as f:
    dump(names, f)
