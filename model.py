#================================================================================
#
# Simple template class for creating a model, which will
# read in the train and test data, and write my predictions
#
#================================================================================
import pandas as pd

class BaseModel(object):

    def __init__(self,train_data_filename= './data/train.csv',test_data_filename='./data/test.csv'):
        """
        Read in the training data in pandas,
        """

        # read the data. Getting everything in RAM this way is not a good idea for
        # large datasets, but we are playing with tens of kb here. 
        self.df_train = pd.read_csv(train_data_filename)
        self.df_test  = pd.read_csv(test_data_filename)


    def train_model(self):
        """
        This routine should be overwritten with a model-specific version.
        
        """

        pass

    def write_prediction(self,prediction_filename = 'prediction.csv'):
        """
        Write prediction to a csv file.
        We assume here that a DataFrame has been created, with the 
        passengerId and Survived columns.
        """

        try:
            # write only two columns, and not the index!
            self.df_prediction.to_csv(prediction_filename,columns=['PassengerId','Survived'],index=False)
        except:
            print 'ERROR: cannot write prediction. Did you train the model?'


