#================================================================================
#
# model which only takes into account gender, fare and class
#
#================================================================================
import pandas as pd
import numpy as N
from model import basemodel 

class genderclassmodel(basemodel):


    def build_survival_table(self):
        """
        Following the tutorial, this function creates a survival table, where
        test case passengers either live or die depending on the majority
        behavior of passengers in the training set corresponding to the same category.
        """

        max_fare = self.df_train['Fare'].max()+100.

        self.genders      = ['male','female']
        self.classes      = [1,2,3]
        self.Fare_limits  = [[0.,10],[10,20],[20,30],[30,max_fare]]

        self.survival_table = N.zeros([4,3,2])

        # loop on the 4x3x2 possibilities, creating different categories of
        # passengers
        for ifare, farel in enumerate(self.Fare_limits):
            cf = ( self.df_train['Fare'] >= farel[0] ) & (self.df_train['Fare'] < farel[1] )

            for ic, cl in enumerate(self.classes):

                cc =  self.df_train['Pclass'] == cl

                for ig, gender in enumerate(self.genders):
                    cg =  self.df_train['Sex'] == gender

                    condition = cf & cc & cg

                    val = self.df_train[condition]['Survived'].values

                    survived = N.sum(val == 1)
                    died     = N.sum(val == 0)
                    total = died+survived

                    if total > 0:
                        rate = 100*(1.*survived)/(1.*total)
                    else:                            
                        rate = 0.

                    if rate > 50.:
                        self.survival_table[ifare,ic,ig] = 1

    def find_fare_index(self,passenger_fare):
        """
        Find the appropriate index for the survival table in the fare dimension.
        """
        found_fare = False

        for i, farel in enumerate(self.Fare_limits):
            if passenger_fare >= farel[0] and passenger_fare < farel[1]:
                ifare = i
                found_fare = True
                break

        if not found_fare:
            ifare = None

        return ifare

    def find_gender_index(self,passenger_gender):
        """
        Find the appropriate index for the survival table in the gender dimension.
        """
        found_gender = False
        for i, gender in enumerate(self.genders):
            if passenger_gender == gender:
                igender = i
                found_gender = True
                break

        if not found_gender:
            igender = None

        return igender

    def find_class_index(self,passenger_class):
        """
        Find the appropriate index for the survival table in the class dimension.
        """
        found_class = False
        for i, cls in enumerate(self.classes):
            if passenger_class == cls:
                iclass = i
                found_class = True
                break

        if not found_class:
            iclass = None

        return iclass

    def test_survival_of_passenger(self,passenger_gender,passenger_class,passenger_fare):
        """
        Identify index of passender in survival table, and return survival state.
        """

        iclass  = self.find_class_index(passenger_class)
        igender = self.find_gender_index(passenger_gender)
        ifare   = self.find_fare_index(passenger_fare)

        if iclass != None and  igender != None and  ifare != None:
            survival = int(self.survival_table[ifare,iclass,igender])
        else:
            survival = 0

        return survival
    
    def train_model(self):
        """
        Overload the base class generic routine by a model-specific
        one. In this case, we look up survival in a table.
        """

        self.build_survival_table()

        # extract series from the test pandas
        ids     = self.df_test['PassengerId']
        genders = self.df_test['Sex']
        fares   = self.df_test['Fare']
        cls     = self.df_test['Pclass']
        
 

        survival = []
        for passenger_gender,passenger_class,passenger_fare in zip(genders,cls,fares):
            passenger_survival = self.test_survival_of_passenger(passenger_gender,passenger_class,passenger_fare)
            survival.append(passenger_survival )


        # create the prediction, passing the data as a dictionary to  create a pandas
        self.df_prediction = pd.DataFrame({'PassengerId':ids,'Survived':survival}) 


if __name__ == "__main__":
    # If this file is invoked directly by python (as opposed to imported in another file)
    # execute what follows 
    model = genderclassmodel()
    model.train_model() 
    model.write_prediction( prediction_filename = 'genderclassmodel.csv')
