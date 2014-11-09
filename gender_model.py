#================================================================================
#
# model which only takes into account gender
#
#================================================================================
import pandas as pd
from model import basemodel 

class genderbasedmodel(basemodel):

    def train_model(self):
        """
        Overload the base class generic routine by a model-specific
        one. In this case, female survive and male snuff it!
        """

        # extract series from the test pandas
        genders = self.df_test['Sex']
        ids     = self.df_test['PassengerId']


        # create a survival series, with men dying and women surviving
        survival = genders.copy() 
        survival[ genders == 'female'] = 1
        survival[ genders == 'male']   = 0

        # create the prediction, passing the data as a dictionary to  create a pandas
        self.df_prediction = pd.DataFrame({'PassengerId':ids,'Survived':survival}) 


        

if __name__ == "__main__":
    # If this file is invoked directly by python (as opposed to imported in another file)
    # execute what follows 
    model = genderbasedmodel()
    model.train_model() 
    model.write_prediction( prediction_filename = 'genderbasedmodel.csv')
