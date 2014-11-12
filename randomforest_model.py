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

    def clean_data(self,df):
        """
        Returns a cleaned data frame, changing some strings into integer.

        note: Ports_dict is hard coded so it is the same for any give data frame
        if a new port appears in a new data sample, just add a new key to Ports_dict
        """

        ## Sex: female = 0, male = 1
        df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

        ## Port
        # Embarked (at port) from 'C', 'Q', 'S'
        # Could be improved (absolute number do not have real meaning here)
        # Replace NA with most frequent value
        # DataFRame.mode() returns the most frequent object in a set
        # here Embarked.mode.values is a numpy.ndarray type (what pandas use to store strings) 
        if len(df.Embarked[ df.Embarked.isnull() ]) > 0:
            most_common_value = df.Embarked.dropna().mode().values[0]
            df.loc[df.Embarked.isnull(),'Embarked'] = most_common_value 

        # The following line produces [(0, 'C'), (1, 'Q'), (2, 'S')]
        #Ports = list(enumerate(np.unique(df['Embarked'])))
        # Create dic {port(char): port(int)}
        #Ports_dict = { name : i for i, name in Ports }
        Ports_dict = {'Q': 1, 'C': 0, 'S': 2}
        # Converting port string as port int
        df.Embarked = df.Embarked.map( lambda x: Ports_dict[x]).astype(int)

        ## Age
        median_age = df['Age'].dropna().median()
        if len(df.Age[ df.Age.isnull() ]) > 0:
            df.loc[ (df.Age.isnull()), 'Age'] = median_age

        ## Fare
        # All the missing Fares -> assume median of their respective class
        if len(df.Fare[ df.Fare.isnull() ]) > 0:
            median_fare = np.zeros(3)
            for f in range(0,3):
                median_fare[f] = df[ df.Pclass == f+1 ]['Fare'].dropna().median()
            for f in range(0,3):
                df.loc[ (df.Fare.isnull()) & (df.Pclass == f+1 ), 'Fare'] = median_fare[f]
        
        return df


    def train_model(self,n_estimators=100):
        """
        Overload the base class generic method with a specific one.
        """
        ###### Data cleanup

        # TRAIN DATA
        # train_df is a data frame
        self.df_train = self.clean_data(self.df_train)

        #removes strings features
        self.df_train = self.df_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

        # TEST DATA
        self.df_test = self.clean_data(self.df_test)
        ids = self.df_test['PassengerId'].values
        self.df_test = self.df_test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

        ###### Convert data frame to numpy array
        train_data = self.df_train.values
        test_data = self.df_test.values

        print 'Training...'
        forest = RandomForestClassifier(n_estimators=n_estimators)
        forest = forest.fit( train_data[0:,1:], train_data[0:,0] )

        print 'Predicting...'
        survival = forest.predict(test_data).astype(int)

        self.df_prediction = pd.DataFrame({'PassengerId':ids,'Survived':survival}) 


if __name__=='__main__':
    model = RandomForestModel()
    model.train_model(n_estimators=10)
    model.write_prediction( prediction_filename = 'RandomForestModel.csv')
