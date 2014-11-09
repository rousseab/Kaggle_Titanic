
#================================================================================
# Play a bit and visualize the data within the file
#
#================================================================================
import matplotlib.pyplot as plt

import numpy as N
import pandas as pd

# The Kaggle data has been downloaded to a local directory 
train_data_filepath = './data/train.csv'

# the variable data will contain a panda DataFrame
df = pd.read_csv(train_data_filepath )


# Let's visualize survival by genders
# - This corresponds to the first part of the "Getting Started with Excel" tutorial
# - To use the pandas plotting facilities (which are just wrappers around matplotlib)
#   we must build a dataframe which contains the data in the right format

genders = ['male','female']
survival_dict = {}
for gender in genders:

    # extract the survival rating for each passenger of a given gender
    val = df[ df['Sex'] == gender ]['Survived'].values

    survival_dict[gender] = {'Died': N.sum(val == 0), 'Survived':N.sum(val == 1) }


df_survival = pd.DataFrame(survival_dict)
df_survival.plot(kind='bar', stacked=True,title='Survival of the Titanic disaster, by gender')


plt.show()
