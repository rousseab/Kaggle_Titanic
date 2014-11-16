#================================================================================
#
# Random forest model, which trains on many columns of the dat.
# This code was initially written by JF Rajotte, and modified to fit here.
#================================================================================


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from model import BaseModel 


class RandomForestModel(BaseModel):

    def clean_data_gender(self,df):
        """
        Returns a cleaned data frame, where the string associated to
        gender is replaced by 0 (female) and 1 ( male).

        Correction is in place, without creating a new column.
        """
        ## Sex: female = 0, male = 1
        gender_dict = {'female': 0, 'male': 1}
        df['Sex']   = df['Sex'].map(gender_dict).astype(int)

    def clean_data_port(self,df):
        """
        Returns a cleaned data frame, where the string associated to
        the port of embarkment is replaced by an integer.

        The ports are labelled 'C', 'Q', 'S'.

        """
        # Replace NA with most frequent value
        # DataFRame.mode() returns the most frequent object in a set
        # here Embarked.mode.values is a numpy.ndarray type (what pandas use to store strings) 
        if len(df.Embarked[ df.Embarked.isnull() ]) > 0:
            most_common_value = df.Embarked.dropna().mode().values[0]
            df.loc[df.Embarked.isnull(),'Embarked'] = most_common_value 

        Ports_dict = {'Q': 1, 'C': 0, 'S': 2}
        # Converting port string as port int
        df['Embarked'] = df['Embarked'].map(Ports_dict).astype(int)

    def clean_data_age(self,df):
        """
        Returns a cleaned data frame, where the missing age 
        information is "guessed".
        """
        # Following the python turotials on the Kaggle web site, let's
        # replace the missing age information with the median value  
        # in the corresponding (gender,class) group.

        # extract the gender and class labels
        Genders = np.unique(df['Sex'].values)
        Classes = np.unique(df['Pclass'].values)

        median_age_dictionary = {}

        for g in Genders:
            for c in Classes:
                # compute the median age for this subgroup of passengers, removing NA values 
                median_age = df[(df['Sex'] == g) & (df['Pclass'] == c)]['Age'].dropna().median()
                # populate dictionary with values
                median_age_dictionary[(g,c)] = median_age 

        # Fill-in the missing age values according to the gender+class-based median values
        if len(df.Age[ df.Age.isnull() ]) > 0:
            for key, median_age in median_age_dictionary.iteritems(): 
                g = key[0]
                c = key[1]
                df.loc[ (df.Age.isnull()) & (df.Sex == g) & (df.Pclass == c),'Age'] = median_age


    def clean_data_fare(self,df):
        """
        Returns a cleaned data frame, where the missing fares
        information is "guessed".
        """
        # let's replace the missing Fare information with the median value  
        # in the corresponding class group.

        # extract the class labels
        Classes = np.unique(df['Pclass'].values)

        median_fare_dictionary = {}

        for c in Classes:
            # compute the median age for this subgroup of passengers, removing NA values 
            median_fare = df[ df['Pclass'] == c]['Fare'].dropna().median()
            # populate dictionary with values
            median_fare_dictionary[c] = median_fare

        # All the missing Fares -> assume median of their respective class
        if len(df.Fare[ df.Fare.isnull() ]) > 0:
            for c, median_fare in median_fare_dictionary.iteritems(): 
                df.loc[ (df.Fare.isnull()) & (df.Pclass == c),'Fare'] = median_fare

                
    def clean_data(self,df):
        """
        Returns a cleaned data frame, changing some strings into integer.
        """
        ## Sex: female = 0, male = 1
        self.clean_data_gender(df)

        ## Port
        self.clean_data_port(df)

        ## Age
        self.clean_data_age(df)

        ## Fare
        self.clean_data_fare(df)

        
        return df


    def train_model(self,n_estimators=100):
        """
        Overload the base class generic method with a specific one.
        """
        ###### Data cleanup

        # TRAIN DATA
        # train_df is a data frame
        self.df_train = self.clean_data(self.df_train)

        #remove data which cannot (or should not) be used to train
        self.df_train = self.df_train.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

        # TEST DATA
        self.df_test = self.clean_data(self.df_test)
        ids = self.df_test['PassengerId'].values
        self.df_test = self.df_test.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

        ###### Convert data frame to numpy array
        train_data = self.df_train.values
        test_data = self.df_test.values

        print 'Training...'
        forest = RandomForestClassifier(n_estimators=n_estimators)
        forest = forest.fit( train_data[:,1:], train_data[:,0] )

        print 'Predicting...'
        survival = forest.predict(test_data).astype(int)

        self.df_prediction = pd.DataFrame({'PassengerId':ids,'Survived':survival}) 


if __name__=='__main__':
    model = RandomForestModel()
    model.train_model(n_estimators=10)
    model.write_prediction( prediction_filename = 'RandomForestModel.csv')
