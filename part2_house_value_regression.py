import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, ParameterSampler
# import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'


class Regressor():

    def __init__(self, x, nb_epoch = 90, batch_size = 4, learning_rate = 0.02, no_neurons = [100, 200, 20], activation_funs=["tanh","relu","relu"], loss_fn = nn.CrossEntropyLoss()):
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
        self.lb = preprocessing.LabelBinarizer()
        self.scaler_x = preprocessing.MinMaxScaler()
        self.scaler_y = preprocessing.MinMaxScaler()
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.no_neurons = no_neurons
        self.activation_funs = activation_funs
        self.loss_fn = loss_fn
        layers = []
        prev_size = self.input_size
        for size, act_fun in zip(self.no_neurons, self.activation_funs):
            layers.append(nn.Linear(prev_size, size))
            if(act_fun == "relu"):
                layers.append(nn.ReLU())
            elif(act_fun == "sigmoid"):
                layers.append(nn.Sigmoid())
            elif(act_fun == "tanh"):
                layers.append(nn.Tanh())
            prev_size = size
        layers.append(nn.Linear(prev_size, self.output_size))
        self.net = nn.Sequential(*layers)

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

        # Fill nan values with the next valid value for x and y
        x = x.fillna(method = "backfill")
        if isinstance(y, pd.DataFrame):
            y = y.fillna(method="backfill")

        # save the parameters for one hot encoding for column ocean_proximity
        if training:
            self.lb.fit(x.loc[:,"ocean_proximity"])

        # replace ocean_proximity column with one hot encoding vectors
        one_hot_encoding_vectors = self.lb.transform(x.loc[:,"ocean_proximity"])
        x = x.drop(columns = ["ocean_proximity"])
        one_hot_encoding_df = pd.DataFrame(one_hot_encoding_vectors, columns=self.lb.classes_, index=x.index)
        x = x.join(one_hot_encoding_df)

        # save the parameters for min-max scalers
        if training: 
            self.scaler_x.fit(x)
            if isinstance(y, pd.DataFrame):
                self.scaler_y.fit(y)

        # normalise x and y using scalers
        x = self.scaler_x.transform(x)
        if isinstance(y, pd.DataFrame):
            y_arr = self.scaler_y.transform(y)

        return torch.tensor(x).float(), (torch.tensor(y_arr).float() if isinstance(y, pd.DataFrame) else None)


    def fit(self, x, y, final_fit=False, x_train=None, y_train=None, x_test=None , y_test=None):
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
        final_fit_error_train = []
        final_fit_error_test = []
        X_tensor, Y_tensor = self._preprocessor(x, y = y, training = True) 
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate)

        datasets = torch.utils.data.TensorDataset(X_tensor, Y_tensor)    
        train_iter = torch.utils.data.DataLoader(datasets, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.nb_epoch):
            for X_tensor_batch, Y_tensor_batch in train_iter:
                optimizer.zero_grad()
                outputs = self.net(X_tensor_batch)
                loss = criterion(outputs, Y_tensor_batch)
                loss.backward()
                optimizer.step()
            # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, self.nb_epoch, loss.item()))
            if(final_fit):
                final_fit_error_train.append(self.score(x_train, y_train))
                final_fit_error_test.append(self.score(x_test, y_test))

        if(final_fit):
            return(self, final_fit_error_train, final_fit_error_test)

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
        X_tensor, _ = self._preprocessor(x, training = False) # Do not forget
        with torch.no_grad():
            preds = self.net(X_tensor)
        return(preds.detach().numpy())
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
        preds = self.predict(x)
        preds_tensor = torch.tensor(preds, dtype=torch.float)
        _, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        
        preds_rescaled = self.scaler_y.inverse_transform(preds_tensor.detach().numpy())
        Y_rescaled = self.scaler_y.inverse_transform(Y)
        loss = np.sqrt(mean_squared_error(preds_rescaled, Y_rescaled))

        return(loss)
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



def RegressorHyperParameterSearch(x_train, y_train, x_val, y_val): 
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
    best_params = {}
    param_vals = {}
    best_error = 1e10

    param_vals["nb_epoch"] = [30, 50 , 100]
    param_vals["batch_size"] = [8, 128, 1024]
    param_vals["learning_rate"] = [0.01, 0.001, 0.0001]
    param_vals["no_neurons"] = [[50, 10], [6, 8], [50, 30, 10], [100, 200, 20], [30, 60, 40, 20], [100, 200, 50, 20]]
    param_vals["activation_funs"] = [["relu","relu"],["sigmoid","relu"],["tanh","relu"],["sigmoid","relu","relu"],["relu","relu","relu"],["tanh","relu","relu"],["relu","relu","relu","relu"],["tanh","tanh","tanh","tanh"]]
    param_vals["loss_fn"] = [nn.MSELoss(), nn.CrossEntropyLoss()]

    for params in ParameterSampler(param_vals, n_iter = 100, random_state=77):
        if(len(params["no_neurons"]) == len(params["activation_funs"])):
            regressor = Regressor(x_train, **params)
            print(params)
            regressor.fit(x_train, y_train)
            error = regressor.score(x_val, y_val)
            print("\nRegressor error: {}\n".format(error))

            if(error < best_error):
                best_error = error
                best_params = params

    return best_params
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def RegressorHyperParameterSearch_stage2_and_plot(best_params,x_train,y_train,x_val,y_val):

    param_vals = {}
    param_vals["nb_epoch"] = [30, 50, 70, 90, 110, 130, 150, 170]
    param_vals["batch_size"] = [4, 8, 16, 32]
    param_vals["learning_rate"] = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    param_vals["no_neurons"] = [[50,10], [50,30], [30,10], [100, 200, 20], [150,220,30], [80, 150, 15]]

    for param_name in param_vals.keys():
        x_axis = []
        y_axis = []
        best_error = 1e10

        for param in param_vals[param_name]:
            best_params[param_name] = param
            best_params["activation_funs"] = ["relu"] * len(best_params["no_neurons"])
            regressor = Regressor(x_train, **best_params)
            regressor.fit(x_train, y_train)
            error = regressor.score(x_val, y_val)
            y_axis.append(error)
            x_axis.append(str(param))
            if(error < best_error):
                best_error = error
                best_param = param
        
        best_params[param_name] = best_param
        best_params["activation_funs"] = ["relu"] * len(best_params["no_neurons"])
        # plt.plot(x_axis,y_axis)
        # plt.xlabel(param_name)
        # plt.ylabel('loss')
        # plt.title(param_name + ' performance')
        # plt.xticks(rotation=45, ha='right')
        # plt.show()

    return best_params
    



    


def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.2, random_state=77)
    x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size=0.5, random_state=77)

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting

    # Sample model training
    # regressor = Regressor(x_train, nb_epoch = 60, no_neurons = [50,30,10], activation_funs=["sigmoid","relu","relu"])
    # print(regressor.net)
    # regressor.fit(x_train, y_train)

    #######################################################################
    #    ** 1ST STAGE (EXPLORATION STAGE) OF HYPERPARAMETER TUNING **     *
    #######################################################################
    best_params = RegressorHyperParameterSearch(x_train, y_train, x_val, y_val)
    #######################################################################

    #######################################################################
    #    ** 2ND STAGE (EXPLOITATION STAGE) OF HYPERPARAMETER TUNING **    *
    #######################################################################
    # it uses best_params as calculated in the 1st stage
    # best_params = {
    #     "nb_epoch" : 50,
    #     "batch_size" :  8,
    #     "learning_rate" :  0.01,
    #     "no_neurons" :  [100,200, 20],
    #     "activation_funs" :  ["tanh","relu","relu"],
    #     "loss_fn" : nn.CrossEntropyLoss()
    # }
    best_params = RegressorHyperParameterSearch_stage2_and_plot(best_params,x_train,y_train,x_val,y_val)
    #######################################################################


    # Final evaluation of the best model
    regressor = Regressor(x_train)
    regressor,train_error,test_error = regressor.fit(x_train, y_train, final_fit=True, x_train=x_train, y_train=y_train, x_test=x_test , y_test=y_test)

    # plt.scatter(np.arange(len(train_error)),train_error,alpha=0.7)
    # plt.scatter(np.arange(len(test_error)),test_error,alpha=0.7)
    # plt.legend(["Train set", "Test set"])
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training & test set error')
    # plt.show()


    save_regressor(regressor)
    loaded_regressor = load_regressor()

    error = loaded_regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

