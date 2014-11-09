#================================================================================
# Play a bit and visualize the data within the file
#
#================================================================================
import matplotlib.pyplot as plt

import numpy as N
import pandas as pd

# The Kaggle data has been downloaded to a local directory 

# the variable data will contain a panda DataFrame

class visualize(object):
    """
    Let's create a visualization class, so we can reuse code.
    """
    def __init__(self, train_data_filepath = './data/train.csv'):
        """
        Read in the data into a panda.
        """
        self.df = pd.read_csv(train_data_filepath )

    
    def show_survival_by_gender(self):
        """
        Let's visualize survival by genders
        - This corresponds to the first part of the "Getting Started with Excel" tutorial
        - To use the pandas plotting facilities (which are just wrappers around matplotlib)
          we must build a dataframe which contains the data in the right format.
        """

        genders = ['male','female']
        survival_dict = {}
        for gender in genders:
            # extract the survival rating for each passenger of a given gender
            val = self.df[ self.df['Sex'] == gender ]['Survived'].values

            survival_dict[gender] = {'Died': N.sum(val == 0), 'Survived':N.sum(val == 1) }



        df_survival = pd.DataFrame(survival_dict)
        print df_survival 
        df_survival.plot(kind='bar', stacked=True,title='Survival of the Titanic disaster, by gender')


        plt.show()

    def show_survival_by_age_and_gender(self):
        """
        Let's visualize survival by age and genders.
        This corresponds to the second part of the "Getting Started with Excel" tutorial.

        Let's confirm that splitting people between children and adults brings little more info.
        """

        genders = ['male','female']

        adult_age = 18

        adult_survival_dict = {}
        for gender in genders:
            # extract the survival rating for each passenger of a given gender
            val = self.df[ (self.df['Sex'] == gender)*(self.df['Age'] >= adult_age) ]['Survived'].values
            adult_survival_dict[gender] = {'Died': N.sum(val == 0), 'Survived':N.sum(val == 1) }

        children_survival_dict = {}
        for gender in genders:
            # extract the survival rating for each passenger of a given gender
            val = self.df[ (self.df['Sex'] == gender)*(self.df['Age'] < adult_age) ]['Survived'].values
            children_survival_dict[gender] = {'Died': N.sum(val == 0), 'Survived':N.sum(val == 1) }

        adult_df_survival    = pd.DataFrame(adult_survival_dict)
        children_df_survival = pd.DataFrame(children_survival_dict)
        print '\nADULTS'
        print adult_df_survival    
        print '\nChildren'
        print children_df_survival    

        adult_df_survival.plot(kind='bar', stacked=True,title='Survival of the Titanic disaster, for ADULTS by gender')
        children_df_survival.plot(kind='bar', stacked=True,title='Survival of the Titanic disaster, for CHILDREN by gender')


        plt.show()

if __name__ == "__main__":
    V = visualize()
    #V.show_survival_by_gender()
    V.show_survival_by_age_and_gender()
