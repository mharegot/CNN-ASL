fp = open("sign_mnist_train.csv", "r")

data = []
fp.readline()
for line in fp:
    strdata = line.split(",")
    intdata = [int(x) for x in strdata]
    data.append(intdata)

# calculates number of examples for each letter
letters= {}
for i in range(len(data)):
    label = data[i][0]
    if label in letters:
        letters[label] += 1
    else:
        letters[label] = 1

names = []
weights = []
# shows counts for each letter
for i in range(0,25):
    if i!=9:
        names.append(str(i))
        weights.append(letters[i])
        print(i, letters[i])

import texttable as tt
tab = tt.Texttable()
headings = ['Letter','Occurrences']
tab.header(headings)



for row in zip(names,weights):
    tab.add_row(row)

s = tab.draw()
print (s)
