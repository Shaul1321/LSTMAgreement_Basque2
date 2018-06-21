"""
========
Barchart
========

A bar plot with errorbars and height labels on individual bars
"""
import numpy as np
import matplotlib.pyplot as plt

N = 4
with_verb = [80.5, 81.0, 91.4, 94.0]
without_verb = [67.6, 65.8, 86.6, 93.3]

#ek = (80.5, 67.6)
#ak = (81, 65.8)
#a = (91.4, 86.6)
#any_suffix = (94, 93.3)

ind = np.arange(N)  # the x locations for the groups
width = 0.3       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, with_verb, width)


rects2 = ax.bar(ind + width, without_verb, width)


# add some text for labels, title and axes ticks
ax.set_ylabel('Model F1 score (%)')
ax.set_title('Suffix prediction accuracy with and without the presence of the original verb form')
ax.set_xticks(ind + width / 2)
ax.set_yticks(np.arange(0,100,5))
ax.set_xticklabels(('ek (erg. pl.)', 'ak (abs. pl./erg. sg.)', 'a (abs. sg.)', 'any suffix'))
#ax.set_ylim([0,80])

ax.legend((rects1, rects2), ('verb present', 'verb omitted'), loc='best')

plt.show()
