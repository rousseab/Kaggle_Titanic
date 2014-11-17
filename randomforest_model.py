#================================================================================
#
# Random forest model, which trains on many columns of the dat.
# This code was initially written by JF Rajotte, and modified to fit here.
#================================================================================


import pandas as pd
import random
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

    def check_model(self,n_estimators=100,train_fraction=0.7):
        """
        Trains a fraction of the data frame sample
        and checks the performance on the rest of the sample
        Default fraction for training is as suggested by Andrew Ng in his ML course.

        Code inspired from JF Rajotte's implementation.
        """

        # fraction of data used for training
        ntrain = int(train_fraction*len(self.df_train))


        # extract a RANDOM set of indices, of dimension ntrain

        all_indices = np.arange(len(self.df_train))

        train_indices = np.sort(random.sample(all_indices, ntrain))
        test_indices  = np.setdiff1d(all_indices,train_indices )

        # replace the training and testing dataframe by
        # subgroups of the original training set
        # careful! we are overwritting, order matters here

        # note the cryptic idioms; simple slicing doesn't work.
        self.df_test  = self.df_train[ self.df_train.index.isin(test_indices)]

        test_survival = self.df_test.Survived.values

        # algorithm expects the Survived column to be missing from the testing data
        self.df_test = self.df_test.drop(['Survived'], axis=1)

        self.df_train = self.df_train[ self.df_train.index.isin(train_indices)]

        # we can call the original routine, the data has been overwritten to fit 
        # our purpose
        self.train_model(n_estimators)

        # extract the survival info from the prediction
        computed_survival = self.df_prediction.Survived.values

        #Prediction result variables
        true_positive  = 0
        true_negative  = 0
        false_positive = 0
        false_negative = 0

        #Loop over all passagers and fill the prediction results variables
        for ts, cs in zip(test_survival,computed_survival):
            if ts == 1 and cs == 1:
                true_positive  += 1
            elif ts == 1 and cs == 0:
                false_positive  += 1
            elif ts == 0 and cs == 1:
                false_negative  += 1
            elif ts == 0 and cs == 0:
                true_negative += 1

        ##Now compute some stats
        self.accuracy = float(true_positive + true_negative)/float(true_positive + true_negative + false_positive + false_negative)
        self.precision = float(true_positive)/float(true_positive + false_positive)
        self.recall = float(true_positive)/float(true_positive + false_negative)
        self.f1score = 2*self.precision*self.recall/(self.precision + self.recall)

        print '\n\n\n' + 20*'---' + '\nTraining performance summary:\n'
        print 'True positive\t{0}'.format(true_positive)
        print 'True negative\t{0}'.format(true_negative)
        print 'False positive\t{0}'.format(false_positive)
        print 'False negative\t{0}'.format(false_negative)

        print '\nAccuracy:\t{0}'.format(round(self.accuracy, 2))
        print 'Precision:\t{0}'.format(round(self.precision, 2))
        print 'Recall: \t{0}'.format(round(self.recall, 2))
        print 'F1Score:\t{0}'.format(round(self.f1score,2))



if __name__=='__main__':

    n_average = 50

    n_estimators  = 50
    list_train_fraction = np.arange(0.2,0.8,0.025)
    list_x        = []
    list_accuracy = []
    list_precision= []
    list_F1       = []

    for train_fraction in list_train_fraction: 

        accuracy  = 0.
        precision = 0.
        f1score   = 0.
        for i in np.arange(n_average):
            model = RandomForestModel()
            model.check_model(n_estimators=n_estimators, train_fraction=train_fraction)

            accuracy  += model.accuracy
            precision += model.precision
            f1score   += model.f1score 


        list_x.append( len(model.df_train) )

        list_accuracy.append(accuracy/n_average)
        list_precision.append(precision/n_average)
        list_F1.append(f1score/n_average)


    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams['font.size'] = 26.


    kwargs = {'ms':10,'mew':2}

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(list_x,list_accuracy,'gx',label='Accuracy',**kwargs)
    ax.plot(list_x,list_precision,'rs',label='Precision',**kwargs)
    ax.plot(list_x,list_F1,'bo',label='F1 score',**kwargs)

    ax.grid(True,linestyle='-',color='grey',alpha=0.5)
    ax.legend(  loc = 0, fancybox=True,shadow=True,  borderaxespad=0.)
    ax.set_xlabel('number of passengers for training')

    #ax.set_xlabel(
    fig.subplots_adjust(left    =       0.10,
                        bottom  =       0.10,
                        right   =       0.90,
                        top     =       0.90,
                        wspace  =       0.20,
                        hspace  =       0.20)

    
    #plt.set_xlabel('train fracion')
    plt.show()


    #model.train_model(n_estimators=10)
    #model.write_prediction( prediction_filename = 'RandomForestModel.csv')
