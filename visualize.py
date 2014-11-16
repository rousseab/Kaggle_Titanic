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
        for gender in genders:
            survival_rate = df_survival[gender]['Survived']/(1.*df_survival[gender].sum())
            print '%s survival rate: %4.1f '%(gender,100.*survival_rate)


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
        for gender in genders:
            survival_rate = adult_df_survival[gender]['Survived']/(1.*adult_df_survival[gender].sum())
            print '%s survival rate: %4.1f '%(gender,100.*survival_rate)


        print '\nChildren'
        print children_df_survival    
        for gender in genders:
            survival_rate = children_df_survival[gender]['Survived']/(1.*children_df_survival[gender].sum())
            print '%s survival rate: %4.1f '%(gender,100.*survival_rate)

        adult_df_survival.plot(kind='bar', stacked=True,title='Survival of the Titanic disaster, for ADULTS by gender')
        children_df_survival.plot(kind='bar', stacked=True,title='Survival of the Titanic disaster, for CHILDREN by gender')



        plt.show()
        return adult_df_survival    


    def show_survival_by_class_gender_price(self):
        """
        Let's visualize survival by gender, class and price paid for the ticket.

        This corresponds to the third part of the "Getting Started with Excel" tutorial.

        """

        max_fare = self.df['Fare'].max()+1.

        # We create a 2 x 3 x 4 set of limits. We will wish to bin survival and death
        # according to these variables 
        genders      = ['male','female']
        clses        = [1,2,3]
        Fare_limits  = [[0.,10],[10,20],[20,30],[30,max_fare]]

        survival_dic = {}
        df_survival_dic = {}

        print '#Fare limits ($)       Class         Gender       Survived    Died    survival rate (%)'
        print '======================================================================================='
        for farel in Fare_limits:
            cf = ( self.df['Fare'] >= farel[0] ) & (self.df['Fare'] < farel[1] )
            print ' %2.1f -- %2.1f'%(farel[0],farel[1])
            survival_dic[farel[0]] = {}
            df_survival_dic[farel[0]] = {}
            
            for cl in clses:
                cc =  self.df['Pclass'] == cl
                print '                         %i'%cl
                survival_dic[farel[0]][cl] = {}
                df_survival_dic[farel[0]][cl] = {}
                for gender in genders:
                    cg =  self.df['Sex'] == gender

                    condition = cf & cc & cg

                    val = self.df[condition]['Survived'].values

                    survival_dic[farel[0]][cl][gender] = {'Died': N.sum(val == 0), 'Survived':N.sum(val == 1) }
                    if len(survival_dic[farel[0]][cl]) >1:
                        ititle = 'Class:{0}, fare:[{1},{2}]'.format(cl, round(farel[0],0), round(farel[1],0))
                        df_survival_dic[farel[0]][cl]    = pd.DataFrame(survival_dic[farel[0]][cl])
                        df_survival_dic[farel[0]][cl].plot(kind='bar', stacked=True,title=ititle)

                    survived = N.sum(val == 1)
                    died     = N.sum(val == 0)

                    total = died+survived

                    if total > 0:
                        rate = 100*(1.*survived)/(1.*total)
                    else:                            
                        rate = 0.

                    print '                                     %8s         %3i    %3i          %4.1f'%(gender, survived, died,rate)

        plt.show()

if __name__ == "__main__":
    V = visualize()
    #V.show_survival_by_gender()
    #adult_df_survival    = V.show_survival_by_age_and_gender()

    V.show_survival_by_class_gender_price()
