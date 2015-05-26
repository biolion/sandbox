# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # vietnam snake
# ####ieuan.clay@gmail.com
# May 2015
# 
# Inspired by [guardian article](http://www.theguardian.com/science/alexs-adventures-in-numberland/2015/may/20/can-you-do-the-maths-puzzle-for-vietnamese-eight-year-olds-that-has-stumped-parents-and-teachers) on maths problems given to school children in vietnam.

# <codecell>

import itertools
# import random
import numpy as np
import matplotlib.pylab as plt
import scipy
import scipy.cluster.hierarchy as sch

# <markdowncell>

# ## The problem
# [The original problem](http://www.theguardian.com/science/alexs-adventures-in-numberland/2015/may/20/can-you-do-the-maths-puzzle-for-vietnamese-eight-year-olds-that-has-stumped-parents-and-teachers) is posed here.
# <center><img src='http://i.guim.co.uk/static/w-620/h--/q-95/sys-images/Guardian/Pix/pictures/2015/5/20/1432109324993/f7e7f4a5-b59c-4580-88f4-ddc609584d19-bestSizeAvailable.png' width=400px></center>
# 
# You need to fill in the gaps with the digits from 1 to 9 so that the equation makes sense, following the order of operations - multiply first, then division, addition and subtraction last (see also [python operator precedence](https://docs.python.org/3.3/reference/expressions.html#operator-precedence)).

# <markdowncell>

# ## Brute force approach
# Calculate all permutations, and examine answers

# <codecell>

answer = 66

# <codecell>

def get_answer(foo):
    (a,b,c,d,e,f,g,h,i) = foo
    bar = a + 13 * b / c + d + 12 * e - f - 11 + g * h / i - 10
    return(bar)

# <codecell>

# calculate all results
perms = [i for i in itertools.permutations(np.arange(9) +1)]
results = [get_answer(perm) for perm in perms]

# <codecell>

# how many results are intergers?
print(sum([(result % 1 == 0) for result in results]), " / ", len(results), " combinations result in integer answers")

# <codecell>

# what is the distribution of all possible results?
#hist, bin_edges = np.histogram(results, bins=50, density=False)
plt.hist(results, bins=50)
plt.title("All possible results")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
# integer results
plt.hist([result for result in results if result % 1 == 0], bins=50)
plt.title("Integer results")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# <markdowncell>

# Examine results which give right answer

# <codecell>

answers = [perm for (perm, result) in zip(perms, results) if result == answer]

# <codecell>

# how many possible answers?
print(len(answers), " / ", len(perms))
print((len(answers) / len(perms))* 100, "%")

# <codecell>

### any relations between the answers?
# [see docs](http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.spatial.distance.cdist.html)

## identity between the permutations
# create empty ndarray
answer_mat = np.zeros(shape=(len(answers), len(answers)), dtype=np.int)
for (i, permi) in enumerate(answers):
    for (j, permj) in enumerate(answers[(i+1):]):
        answer_mat[i,i+1+j] = sum([(a == b) for a, b in zip(permi, permj)])

# <codecell>

# spot check results
for i in range(len(answers[0])):
    print(sum(answer_mat == i), " answers equal to ", i)
for element in [(i,j) for i in range(len(answer_mat)) for j in range(len(answer_mat))]:
    if answer_mat[element] == 6:
        print(">>>>")
        print(answers[element[0]])
        print(answers[element[1]])

# <codecell>

# plot it
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(answer_mat, interpolation='nearest', cmap=plt.cm.ocean)
plt.colorbar()
plt.show()

# <codecell>

# fill matrix, convert to distance and replot
answer_mat = answer_mat + answer_mat.T
np.fill_diagonal(answer_mat, len(answers[0]))
answer_mat = 1- (answer_mat / (len(answers[0])))
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(answer_mat, interpolation='nearest', cmap=plt.cm.ocean)
plt.colorbar()
plt.show()

# <codecell>

# any clusters?
# Compute and plot dendrogram.
fig = plt.figure()
axdendro = fig.add_axes([0.09,0.1,0.2,0.8])
Y = sch.linkage(answer_mat, method='centroid')
Z = sch.dendrogram(Y, orientation='right')
axdendro.set_xticks([])
axdendro.set_yticks([])

# Plot distance matrix.
axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
index = Z['leaves']
answer_mat = answer_mat[index,:]
answer_mat = answer_mat[:,index]
im = axmatrix.matshow(answer_mat, aspect='auto', origin='lower')
axmatrix.set_xticks([])
axmatrix.set_yticks([])

# Plot colorbar.
axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
pylab.colorbar(im, cax=axcolor)

# Display and save figure.
fig.show()

# <codecell>

# any regions of the equation that seems to go together, manhattan distance on "path"?

# <codecell>


