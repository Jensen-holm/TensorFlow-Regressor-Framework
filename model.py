import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class model():

    def __init__(self, name, dataset, explanatory, response, split = .3, standardize = True):
        self.name = name
        self.split = split # 70-30 train_test_split
        self.standardize = standardize
        self.model = self.build(dataset, explanatory, response)

        # standardize fit branch
        if self.standardize == True:
            self.Fit(self.features_train_scaled, self.labels_train_scaled)
            self.Eval(self.features_test_scaled, self.labels_test_scaled)
        elif self.standardize == False:
            self.Fit(self.features_train, self.labels_train)
            self.Eval(self.features_test, self.labels_test)

    def build(self, dataset, explanatory, response, dummies = True, dropout_ = .3):

        # create instance of sequential model
        my_model = Sequential(name = self.name)
        df = dataset
        # to dummy or not to dummy
        if dummies == True:
            df = pd.get_dummies(pd.DataFrame(dataset))
        elif dummies == False:
            df = pd.DataFrame(dataset)

        # train_test_split
        # can only be one response in sequential models
        self.features_train, self.features_test, self.labels_train, self.labels_test = train_test_split(explanatory, response, test_size = self.split, random_state = 42)

        # to standardize or not to standardize
        if self.standardize == True:

            # standardization and the column transformer
            numeric_features = explanatory.select_dtypes(include = ['float64','int64'])
            numeric_columns = numeric_features.columns

            ct = ColumnTransformer([('only numeric',StandardScaler(), numeric_columns)], remainder = 'passthrough')

            # ct for the lables
            # since there is only one column
            self.labels_train_scaled = (self.labels_train - self.labels_train.mean())/self.labels_train.std() 
            self.labels_test_scaled = (self.labels_test - self.labels_train.mean()) / self.labels_train.std()

            self.features_train_scaled = ct.fit_transform(self.features_train)
            self.features_test_scaled = ct.fit_transform(self.features_test)

            my_model.add(layers.InputLayer(input_shape = (self.features_train_scaled.shape[1])))
            # adding hidden layers
            my_model.add(layers.Dense(32, activation = 'relu'))
            my_model.add(layers.Dropout(dropout_))
            # output layer
            my_model.add(layers.Dense(1))
            # optimiser
            opt = Adam(learning_rate = 0.01)
            my_model.compile(loss = 'mse', metrics = ['mae'], optimizer = opt)
            self.es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience = 20)
            print(my_model.summary())
            return my_model

        elif self.standardize == False:

            # same thing but without standardization and the column transformer
            my_model.add(layers.InputLayer(input_shape = (self.features_train.shape[1])))
            # adding hidden layers
            my_model.add(layers.Dense(32, activation = 'relu'))
            my_model.add(layers.Dropout(dropout_))
            # output layer
            my_model.add(layers.Dense(1))
            # optimiser
            opt = Adam(learning_rate = 0.01)
            my_model.compile(loss = 'mse', metrics = ['mae'], optimizer = opt)
            self.es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 0, patience = 20)
            print(my_model.summary())
            return my_model

        elif self.standardize != False:
            return f"standardize should be a Boolean Value, not {self.standardize}."

    # fit to training data
    def Fit(self, features_train, labels_train, num_epochs = 5, batch_size = 1):
        return self.model.fit(features_train, labels_train, epochs=num_epochs, batch_size= batch_size, verbose=1, validation_split = 0.2, callbacks = [self.es])

    # evaluate on the test data
    def Eval(self, features_test, labels_test):
        val_mse, val_mae = self.model.evaluate(features_test, labels_test, verbose = 0)
        print(f"\n --- RESULTS ---\nMAE: {val_mae}\nMSE: {val_mse}")
