import torch
import pickle
import numpy as np
import pandas as pd
import sys
import torch.optim as optim

pd.options.mode.chained_assignment = None  # default='warn'


class Regressor():

    def __init__(self, x, nb_epoch = 1000):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 


        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # self.layer_l1 = torch.nn.Linear(self.input_size, self.output_size) # hidden->output weights
                self.layer_l1 = torch.nn.Linear(9, 1) # hidden->output weights
                # self.layer_l1 = torch.nn.Linear(9 , 5) # input->hidden weights
                # self.layer_l2 = torch.nn.Linear(5 , 10) # input->hidden weights
                # self.layer_l3 = torch.nn.Linear(10, 1) # hidden->output weights

            def forward(self, x):
                x = self.layer_l1(x) # output 
                # x = torch.sigmoid(self.layer_l1(x)) # hidden layer with tanh activation
                # x = torch.sigmoid(self.layer_l2(x)) # output with sigmoid activation
                # x = self.layer_l3(x) # output 
                return x

        self.net = Net()
        print(self.net)

        # Replace this code with your own

        self.trainset_min = []
        self.trainset_max = []
        self.trainset_mean = []

        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        x_copy = x.copy()

        # Mapping of string values to integers
        ocean_proximity_mapping = { 'INLAND':0, '<1H OCEAN':1, 'NEAR OCEAN':2, 'NEAR BAY':3, 'ISLAND':4 }
        x_copy["ocean_proximity"] = x_copy["ocean_proximity"].map(ocean_proximity_mapping)

        # if using train set save normalization parameters
        if(training):
            self.trainset_min = x_copy.min()
            self.trainset_max = x_copy.max()
            self.trainset_mean = x_copy.mean()

        # Fill any empty values with mean values as calculated using train set
        for col in x_copy.columns:
            x_copy[col].fillna(self.trainset_mean[col], inplace=True)

        # Normalize values using training set's min and max values
        def min_max_norm(x_in,col_id):
            return ((x_in - self.trainset_min[col_id]) / (self.trainset_max[col_id] - self.trainset_min[col_id]))
        for col_id,col_name in enumerate(x_copy.columns):
            x_copy[col_name] = x_copy[col_name].apply(min_max_norm, col_id=col_id)

        # for col in x.columns:
        #     print(col)
        #     print(x[col].unique())
        # print(self.trainset_min)
        # print(self.trainset_max)
        # print(np.where(pd.isnull(x)))
        # print(x["ocean_proximity"].unique())

        # Return preprocessed x and y, return None for y if it was None
        return x_copy, (y if isinstance(y, pd.DataFrame) else None)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        learning_rate = 1e-4

        criterion = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.functional.mse_loss
        # optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate)

        # batch_size = 4
        # trainloader = torch.utils.data.DataLoader(X, batch_size=batch_size,
        #                                   shuffle=True, num_workers=2)
        X_tensor = torch.tensor(X.values, dtype=torch.float)
        Y_tensor = torch.tensor(Y.values, dtype=torch.float)

        
        for epoch in range(self.nb_epoch):
            # running_loss = 0.0

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # print(X_tensor)
            outputs = self.net(X_tensor)
            # print(outputs)
            loss = criterion(outputs, Y_tensor)
            # print(Y_tensor)
            loss.backward()
            optimizer.step()

            # print statistics
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, self.nb_epoch, loss.item()))            
            # running_loss += loss.item()
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            #     running_loss = 0.0




        # for epoch in range(2):  # loop over the dataset multiple times
        #     running_loss = 0.0
        #     for i, data in enumerate(trainloader):
        #         # get the inputs; data is a list of [inputs, labels]
        #         inputs, labels = data

        #         # zero the parameter gradients
        #         optimizer.zero_grad()

        #         # forward + backward + optimize
        #         outputs = self.net(inputs)
        #         loss = criterion(outputs, labels)
        #         loss.backward()
        #         optimizer.step()

        #         # print statistics
        #         running_loss += loss.item()
        #         if i % 2000 == 1999:    # print every 2000 mini-batches
        #             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        #             running_loss = 0.0


        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        pass

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        return 0 # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    # example_main()

    # print(sys.executable)

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # print(data.head())

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]


    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 1000)
    regressor.fit(x_train, y_train)


    X, Y = regressor._preprocessor(x_train, y = y_train, training = True) # Do not forget
    X_tensor = torch.tensor(X.values, dtype=torch.float)
    print(regressor.net(X_tensor))
    print(Y)


    # save_regressor(regressor)

    # # Error
    # error = regressor.score(x_train, y_train)
    # print("\nRegressor error: {}\n".format(error))

