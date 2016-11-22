"""

devname: vdthatte

using this tutorial
https://github.com/justmarkham/DAT4/blob/master/notebooks/08_linear_regression.ipynb


"""

# imports

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# read data into dataframe

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
data.head()

# printing the head shape of the dataframe

# print(data.shape) # shape (rows, columns) of a dataset

"""

# visualize relation between features and response

fig, axs = plt.subplots(1, 3, sharey=True)
data.plot(kind='scatter', x='TV', y='Sales', ax=axs[0], figsize=(16, 8))
data.plot(kind='scatter', x='Radio', y='Sales', ax=axs[1])
data.plot(kind='scatter', x='Newspaper', y='Sales', ax=axs[2])
plt.show()

"""

# create a fitter model

lm = smf.ols(formula='Sales ~ TV', data=data).fit()

# print the coefficients
# print(lm.params)

# create a dataframe for the statsmodel formula interface

# X_new = pd.DataFrame({'TV':[50]})
# X_new.head()


# print(lm.predict(X_new))

# create a dataframe with min and max values of TV

X_new = pd.DataFrame({'TV': [data.TV.min(), data.TV.max()]})
X_new.head()

preds = lm.predict(X_new)

# print(preds)


# first, plot the observed data
data.plot(kind='scatter', x='TV', y='Sales')

# then, plot the least squares line
plt.plot(X_new, preds, c='red', linewidth=2)

# plt.show()

# print the confidence intervals for the model coefficients
# print(lm.conf_int())

# hypothesis testing and p-values
# print(lm.pvalues)





