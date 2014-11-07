
#================================================================================
# Play a bit and visualize the data within the file
#
#================================================================================
import matplotlib.pyplot as plt
import pandas as pd

# The Kaggle data has been downloaded to a local directory 
train_data_filepath = './data/train.csv'

# the variable data will contain a panda DataFrame
df = pd.read_csv(train_data_filepath )

# Let's plot some data

df_gender = df[['Sex','Survived']]

axes = df_gender.hist(by='Sex')

for ax in axes:
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Dead','Alive'])

plt.show()
