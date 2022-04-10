import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import Series, DataFrame
from sklearn import datasets, svm, tree
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, GridSearchCV
/ usr / local / lib / python2
.7 / dist - packages / pandas / core / computation / __init__.py: 18: UserWarning: The
installed
version
of
numexpr
2.4
.3 is not supported in pandas and will
be
not be
used
The
minimum
supported
version is 2.4
.6

ver = ver, min_ver = _MIN_NUMEXPR_VERSION), UserWarning)
/ usr / local / lib / python2
.7 / dist - packages / matplotlib / __init__.py: 913: UserWarning: axes.color_cycle is deprecated and replaced
with axes.prop_cycle; please use the latter.
warnings.warn(self.msg_depr % (key, alt_key))
In[2]:
# Reading csv file
    df = pd.read_csv('train.csv')
num_rows = df.shape[0]
y = df['price_range']
del df['price_range']
In[3]:
# Scaling the input features
x = df.ix[:, :-1].values
standard_scaler = StandardScaler()
x_std = standard_scaler.fit_transform(x)
        / usr / local / lib / python2
.7 / dist - packages / ipykernel_launcher.py: 2: DeprecationWarning:
.ix is deprecated.Please
use
    .loc
for label based indexing or
.iloc for positional indexing

See the documentation here:
    http: // pandas.pydata.org / pandas - docs / stable / indexing.html
# deprecate_ix

In[4]:
# Splitting input data into train set and test set
X_train, X_test, y_train, y_test = train_test_split(x_std, y, test_size=0.2)
In[5]:
# Classifier
svc = svm.SVC()
parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
clf = GridSearchCV(svc, parameters)
model = clf.fit(X_train, y_train)
In[6]:
pred = clf.predict(X_test)
plt.scatter(y_test, pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
print("Score:", model.score(X_test, y_test))
('Score:', 0.95499999999999996)
In[18]:
plt.figure(figsize=(1000, 800))
sns.heatmap(DataFrame(x_std).corr(), annot=True)
# ,cmap='cubehelix_r')
plt.show()
- --------------------------------------------------------------------------
ValueError
Traceback(most
recent
call
last)
< ipython - input - 18 - 1
b45f2e6389a > in < module > ()
1
plt.figure(figsize=(1000, 800))
- ---> 2
sns.heatmap(DataFrame(x_std).corr(), annot=True)
# ,cmap='cubehelix_r')
3
plt.show()

/ usr / lib / python2
.7 / dist - packages / seaborn / matrix.pyc in heatmap(data, vmin, vmax, cmap, center, robust, annot, fmt, annot_kws,
                                                       linewidths, linecolor, cbar, cbar_kws, cbar_ax, square, ax,
                                                       xticklabels, yticklabels, mask, **kwargs)
444
if square:
    445
ax.set_aspect("equal")
--> 446
plotter.plot(ax, cbar_ax, kwargs)
447
return ax
448

/ usr / lib / python2
.7 / dist - packages / seaborn / matrix.pyc in plot(self, ax, cax, kws)
223
224  # Possibly rotate them if they overlap
--> 225
plt.draw()
226
if axis_ticklabels_overlap(xtl):
    227
    plt.setp(xtl, rotation="vertical")

/ usr / local / lib / python2
.7 / dist - packages / matplotlib / pyplot.pyc in draw()
689
fig.canvas.draw_idle()
690

In [ ]:
#Output file calculations

dfd = pd.read_csv('test.csv')
y = dfd.ix[:,:-1].values
standard_scaler1 = StandardScaler()
x_std1 = standard_scaler1.fit_transform(y)
prediction = clf.predict(x_std1)
np.savetxt('./GridSearchCV.csv',prediction, delimiter=',')
