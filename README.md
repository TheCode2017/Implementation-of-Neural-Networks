# Implementation-of-Neural-Networks
Implementation of neural networks from scratch without using any packages.



In this assignment you will implement the backpropagation algorithm for Neural Networks. and
test it on various real world datasets. Before running the algorithm, you will have to ensure that
all the features of the dataset are properly scaled, standardized, and categorical variables are
encoded using numerical values.
The two steps of this assignment are listed below. You have to create separate class for each of
the steps.
1. Pre-processing:
Pre-processing involves checking the dataset for null or missing values, cleansing the dataset of
any wrong values, standardizing the features and converting any nominal (or categorical)
variables to numerical form. This step is essential before running neural net algorithm, as they
can only accept numeric data and work best with scaled data.
The arguments to this part will be:
- complete input path of the raw dataset
- complete output path of the pre-processed dataset
Your pre-processing code will read in a dataset specified using the first command line argument
and first check for any null or missing values. You will remove any data points (i.e. rows) that
have missing or incomplete features.
Then, you will perform the following for each of the features (independent variables):
- If the value is numeric, it needs to be standardized, which means subtracting the mean from
each of the values and dividing by the standard deviation.
See here for more details: https://en.wikipedia.org/wiki/Feature_scaling#Standardization
- If the value is categorical or nominal, it needs to be converted to numerical values.
For example, if the attribute is gender and is encoded as "male" or "female", it needs to be
converted to 0 or 1. You are free to figure out specifics of your encoding strategy, but be sure to
mention it in the report.
For the output or predicted variable, if its value is numerical or continuous, you are free to
convert it into categorical or binary variables (classification problem) or keep it as it is
(regression problem). You have to specify this clearly in your report.
After completing these steps, you need to save the processed dataset to the path specified by
the second command line argument and use it for the next step.
2. Training a Neural Net:
You will use the processed dataset to build a neural net. The input parameters to the neural net
are as follows:
- input dataset – complete path of the post-processed input dataset
- training percent – percentage of the dataset to be used for training
- maximum_iterations – Maximum number of iterations that your algorithm will run. This
parameter is used so that your program terminates in a reasonable time.
- number of hidden layers
- number of neurons in each hidden layer
For example, input parameters could be:
ds1 80 200 2 4 2
The above would imply that the dataset is ds1, the percent of the dataset to be used for
training is 80%, the maximum number of iterations is 200, and there are 2 hidden layers with
(4, 2) neurons. Your program would have to initialize the weights randomly. Remember to take
care of the bias term (w0) also.
While coding the neural network, you can make the following assumptions:
- the activation function will be sigmoid
- you can use the backpropagation algorithm described in class and presented in the textbook
- the training data will be randomly sampled from the dataset. The remaining will form the test
dataset
- you can use the mean square error as the error metric
- one iteration involves a forward and backward pass of the back propagation algorithm
- your algorithm will terminate when either the error becomes 0 or the max number of
iterations is reached.
After building the model, you will output the model parameters as below:
Layer 0 (Input Layer):
Neuron1 weights:
Neuron 2 weights:
..
Layer 1 (1st hidden layer):
Neuron1 weights:
Neuron 2 weights:
..
Layer n (Last hidden layer):
Neuron1 weights:
Neuron 2 weights:
..
Total training error = ….
You will also apply the model on the test data and report the test error:
Total test error = ….
Testing your program
You will test both parts of your program, first the pre-processing part and then the model
creation and evaluation, on the following datasets:
1. Car Evaluation Dataset
https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
2. Iris dataset
https://archive.ics.uci.edu/ml/datasets/Iris
3. Adult Census Income dataset
https://archive.ics.uci.edu/ml/datasets/Census+Income
For each of the above 3, you have to run your code and report the analysis and results in a
separate folder. The analysis should contain your pre-processing strategy, your best set of
parameters, and the best results in terms of accuracy. You should also keep a log of all your
experiments, which should list how many different combination of parameters you tested.
