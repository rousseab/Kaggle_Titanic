
#================================================================================
# Play a bit and visualize the data within the file
#
#================================================================================
import matplotlib.pyplot as plt
import pandas as pd

# The Kaggle data has been downloaded to a local directory 
train_data_filepath = './data/train.csv'

data = pd.read_csv(train_data_filepath )

# Let's plot some data

gender   = data['Sex']
survival = data['Survived']

#plots = pd.scatter_matrix(data)


#plt.show()
