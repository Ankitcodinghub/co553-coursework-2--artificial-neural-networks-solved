# co553-coursework-2--artificial-neural-networks-solved
**TO GET THIS SOLUTION VISIT:** [CO553 Coursework 2- Artificial Neural Networks Solved](https://www.ankitcodinghub.com/product/co553-coursework-2-artificial-neural-networks-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;96013&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CO553 Coursework 2- Artificial Neural Networks Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
&nbsp;

1 Overview

Please read this manual THOROUGHLY as it contains crucial information about the assign- ment, as well as the support provided.

In this assignment, you will learn to implement and optimise neural network models and apply them to solve a regression task. You are expected to submit a report (up to 5 pages) answering any questions specified and discussing your implementation and the results of your experiments. You should also submit the SHA1 token for the specific commit on your GitLab repository that you wish to be assessed.

This assignment is split into 2 parts:

• Part 1: Create a neural network mini-library. You will create a low-level im- plementation of a multi-layered neural network, including a basic implementation of the backpropagation algorithm. The task also involves implementing necessary functions for data preprocessing, training and evaluation.

• Part 2: Create and train a neural network for regression. You will develop and optimise a neural network architecture to predict the price of houses in California using the California House Prices Dataset. For this task, you will have the choice to either use PyTorch or the mini-library you have just developed.

The mini-library in Part 1 must be designed using NumPy, and will require you to implement a linear layer class, activation functions, a multi-layer network class, a trainer class, and a data preprocessing class. For this part, you may not use libraries that implement automatic differentiation, such as PyTorch or TensorFlow, but you may use basic python libraries (os, sys, math, etc.). This task will be assessed based on the results of corresponding LabTS tests and you do not need to include information about Part 1 in your report.

For the models in Part 2 you must use either the PyTorch neural network library or the mini- library that you developed in the first part. This part of the assignment will be primarily assessed on the basis of your report, where you describe your model, evaluation setup, the hyperparameter tuning process and the final results. There are also LabTS tests for Part 2 which will contribute to the final coursework score.

</div>
</div>
<div class="layoutArea">
<div class="column">
1

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
2 Guidelines and setup

The link to your group GitLab repository with the skeleton code will be available on LabTS (https://teaching.doc.ic.ac.uk/labts/). Using Git will make it easier to collaborate with team members and keep backups of different code versions. We recommend you take advantage of it. You can find a quick introductory handbook to Git in the GitHub documentation: https://guides.github.com/introduction/git-handbook/.

You should implement all your code in Python 3, we will not provide support for other languages. Your code will be assessed and tested on LabTS. The python environment on the LabTS servers closely mimics the ones on the lab workstations. This is a good reason for you to work on the lab workstations!

We will provide a suite of tests for you to test your code prior to submission using LabTS. To test your code, push to your GitLab group repository and access the tests through the LabTS portal. We highly recommend that you test your code on LabTS as soon as possible and throughout the project, to avoid discovering incompatibility issues with the test environment at the last minute. You can test your code on LabTS as often as you need to, and we will consider for marking only the commit corresponding to the SHA1 token you submitted on CATE. These tests aim to guide you in your implementation, not to provide you with an extended testing environment. Thus, do not expect them to give you detailed error messages, and make sure to double-check also components or functions that are not directly tested by them. The available tests are only examples and the final assessment will be performed using a different test suite.

2.1 Working on DoC lab workstations (recommended)

The python3 environment in the Ubuntu workstations in the lab should have everything you need to complete this coursework. Therefore, we recommend you to work on the lab machines from home using SSH. You can find the list of the lab workstation on https://www.doc.ic. ac.uk/csg/facilities/lab/workstations.

It is recommended to use python3 exclusively, the version in lab machines is 3.8.5. In order to load all the packages that you might need for the coursework, run the following command:

export PYTHONUSERBASE=/vol/lab/ml/mlenv

If you don’t want to type that command every time that you connect to your lab machine, you

can add it to your bashrc, so every terminal you open will have that environment variable set:

<pre>echo "export PYTHONUSERBASE=/vol/lab/ml/mlenv" &gt;&gt; ~/.bashrc
</pre>
To test the configuration:

<pre>python3 -c "import numpy as np; import torch; print(np); print(torch)"
</pre>
This should print:

<pre>&lt;module 'numpy' from '/vol/lab/ml/mlenv/lib/python3.8/site-packages/numpy/__init__.py'&gt;
&lt;module 'torch' from '/vol/lab/ml/mlenv/lib/python3.8/site-packages/torch/__init__.py'&gt;
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
2

</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
2.2 Working on your own system

If you decide to work locally on your machine, then you must make sure that your code also runs on the lab machines using the above environment. Anything we cannot run will result in marks for the question being reduced by 30%.

All provided code has been tested on Python 3.8.5. Make sure to install it on your local machine, otherwise you might encounter errors! On Mac OSX, you can use Homebrew to brew install python3.

Part 1: Create a neural network mini-library

In this part, you will implement your own modular neural network mini-library using NumPy. This will require you to complete a number of classes in src/part1_nn_lib.py. This part of the coursework will be evaluated based on your code only using LabTS tests. However, keep in mind that in addition to the public tests you can run by yourself on LabTS, the final assessment will be performed using a different set of private tests. Thus, make sure to double-check also components or functions that are not directly tested by the public tests.

A simple dataset (https://en.wikipedia.org/wiki/Iris_flower_data_set, given in the Gitlab repos- itory as iris.dat) and sample code demonstrating intended usage is provided for debugging purposes as you work through this part of the coursework.

Q1.1: Implement a linear layer:

For this question, you must implement a linear layer (which performs an affine transformation XW +B on a batch of inputs X) by completing the following methods of the LinearLayer class:

Constructor: In the constructor, initialise the attributes you need for this class. In particular, NumPy arrays representing the learnable parameters of the layer, initialized in a sensible manner (hint: you can use the provided xavier_init function). Use the attributes _W, _b to refer to your weights matrix and bias respectively.

Forward pass method: Implement the forward method to do the following:

<ul>
<li>Return the outputs of the layer given a NumPy array representing a batch of inputs.</li>
<li>Store any data necessary for computing the gradients when later performing the backward pass in the _cache_current attribute.</li>
<li>You are NOT allowed to use Python loops in this method (for efficiency reasons); use vectorized operations instead.
Backward pass method: Implement the backward method to do the following:

<ul>
<li>Given the gradient of some scalar function with respect to the outputs of the layer as input, compute the gradient of the function with respect to the parameters of the layer and store them in the relevant attributes defined in the constructor (_grad_W_current and _grad_b_current).</li>
<li>Compute and return the gradient of the function with respect to the inputs of the layer.</li>
<li>You are NOT allowed to use Python loops in this method (for efficiency reasons); use vectorized operations instead.</li>
</ul>
</li>
</ul>
</div>
</div>
<div class="layoutArea">
<div class="column">
3

</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="layoutArea">
<div class="column">
Parameter update method: Implement the update_params method to perform one step of gradient descent on the parameters of the layer (using the stored gradients and the learning rate provided as argument).

Here is an example of how this class can be used:

layer = LinearLayer(n_in=3, n_out=42) # `inputs` shape: (batch_size, 3)

# `outputs` shape: (batch_size, 42) outputs = layer(inputs)

# `grad_loss_wrt_outputs` shape: (batch_size, 42)

# `grad_loss_wrt_inputs` shape: (batch_size, 3) grad_loss_wrt_inputs = layer.backward(grad_loss_wrt_outputs) layer.update_params(learning_rate)

Q1.2: Implement activation function classes:

For this question, you must implement the SigmoidLayer and ReluLayer activation function classes (note: linear activation can be achieved without an activation class). For each of these two classes, complete the following:

Forward pass method: Implement the forward method to do the following:

<ul>
<li>Returns the element-wise transformation of the inputs using the activation function.</li>
<li>Store any data necessary for computing the gradients when later performing the backward pass in the _cache_current attribute.</li>
<li>You are NOT allowed to use Python loops in this method (for efficiency reasons); use vectorized operations instead.
Backward pass method: Implement the backward method to do the following:

<ul>
<li>Compute and return the gradient of the function with respect to the inputs of the layer.</li>
<li>You are NOT allowed to use Python loops in this method (for efficiency reasons); use vectorized operations instead.
Q1.3: Implement a multi-layer network:

For this question, you must implement a multi-layer network (consisting of stacked linear layers and activation functions) by completing the following methods of the MultiLayerNetwork class:

Constructor: Define the following in the constructor:

• Attribute/s containing instances of the LinearLayer class and activation classes, as specified by

the arguments:

– input_dim: an integer specifying the number of input neurons in the first linear layer,

– neurons: a list specifying the number of output neurons in each linear layer, the length of the list determines the number of linear layers,
</li>
</ul>
</li>
</ul>
</div>
</div>
<div class="layoutArea">
<div class="column">
4

</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="layoutArea">
<div class="column">
– activations: a list of activation functions to apply to the output of each linear layer. • Store your layer instances in the _layers attribute.

Forward pass method: Implement the forward method to do the following:

• Return the outputs of the network given a NumPy array representing a batch of inputs (note: the instances of the LinearLayer classes created in the constructor should automatically handle any storage of data needed to compute gradients).

Backward pass method: Implement the backward method to do the following:

<ul>
<li>Given the gradient of some scalar function with respect to the outputs of the network as input, compute the gradient of the function with respect to the parameters of the network. (note: the instances of the LinearLayer classes created in the constructor should automatically handle any storage of computed gradients).</li>
<li>Return the gradient of the function with respect to the inputs of the network.
Parameter update method: Implement the update_params method to perform one step of gradient descent on the parameters of the network (using the stored gradients and the learning rate provided as argument).

Here is an example of how this class can be used:

<pre># The following command will create a MultiLayerNetwork object
# consisting of the following stack of layers:
</pre>
</li>
</ul>
<ul>
<li>
<pre># &nbsp;  - LinearLayer(4, 16)
</pre>
</li>
<li>
<pre># &nbsp;  - ReluLayer()
</pre>
</li>
<li>
<pre># &nbsp;  - LinearLayer(16, 2)
</pre>
</li>
<li># &nbsp;– SigmoidLayer()

network = MultiLayerNetwork(

<pre>    input_dim=4, neurons=[16, 2], activations=["relu", "sigmoid"]
)
</pre>
# `inputs` shape: (batch_size, 4)

# `outputs` shape: (batch_size, 2)

outputs = network(inputs)

# `grad_loss_wrt_outputs` shape: (batch_size, 2)

# `grad_loss_wrt_inputs` shape: (batch_size, 4) grad_loss_wrt_inputs = network.backward(grad_loss_wrt_outputs) network.update_params(learning_rate)

Q1.4: Implement a trainer:

For this question, you must implement a ”Trainer” class which handles data shuffling, training a given network using minibatch gradient descent on a given training dataset, as well as computing the loss on a validation dataset. To do so, complete the following methods of the Trainer class:

Constructor: Define the following in the constructor:

• An attribute (_loss_layer) referencing an instance of a loss layer class as specified by the loss_fun argument (which can take values “mse” or “cross_entropy”, corresponding to the mean-squared error and binary cross-entropy losses respectively.).
</li>
</ul>
</div>
</div>
<div class="layoutArea">
<div class="column">
5

</div>
</div>
</div>
<div class="page" title="Page 6">
<div class="layoutArea">
<div class="column">
Data shuffling: Implement the shuffle method to return a randomly reordered version of the data observations provided as arguments.

Main training loop: Implement the train method to carry out the training loop for the network. It should loop over the following nb_epoch times:

• If shuffle_flag = True, shuffle the provided dataset using the shuffle method.

• Split the provided dataset into minibatches of size batch_size and train the network using

minibatch gradient descent.

Computing evaluation loss: Implement the eval_loss method to compute and return the loss on the provided evaluation dataset.

Here is an example of how this class can be used:

<pre>trainer = Trainer(
    network=network,
    batch_size=32,
    nb_epoch=10,
    learning_rate=1.0e-3,
    shuffle_flag=True,
    loss_fun="mse",
</pre>
<pre>)
trainer.train(train_inputs, train_targets)
print("Validation loss = ", trainer.eval_loss(val_inputs, val_targets))
</pre>
Q1.5: Implement a preprocessor:

Data normalization can be crucial for effectively training neural networks. For this question, you will need to implement a “Preprocessor” class which performs min-max scaling such that the data is scaled to lie in the interval [0, 1]. Same as before, complete the methods of the Preprocessor class.

Constructor: Should compute and store data normalization parameters according to the provided dataset. This function does not modify the provided dataset.

Apply method: Complete the apply method such that it applies the pre-processing operations to the provided dataset and returns the preprocessed version.

Revert method: Complete the revert method such that it reverts the pre-processing operations that have been applied to the provided dataset and returns the reverted dataset. For any dataset A, prep.revert(prep.apply(A)) should return A.

Here is an example of how this class can be used:

<pre>prep = Preprocessor(dataset)
normalized_dataset = prep.apply(dataset)
original_dataset = prep.revert(normalized_dataset)
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
6

</div>
</div>
</div>
<div class="page" title="Page 7">
<div class="layoutArea">
<div class="column">
Part 2: Create and train a neural network for regression

In this part, you will work on the California House Prices Dataset 1. It consists of a publicly available dataset of all the block groups in California from the 1990 Census. A block group is the smallest geographical unit for which the US Census Bureau publishes sample data, and on average it includes 1425.5 individuals living in a geographically compact area. The dataset contains 20,640 observations on ten variables:

1. longitude: longitude of the block group

2. latitude: latitude of the block group

3. housing median age: median age of the individuals living in the block group 4. total rooms: total number of rooms in the block group

5. total bedrooms: total number of bedrooms in the block group

6. population: total population of the block group

7. households: number of households in the block group

8. median income: median income of the households comprise in the block group 9. ocean proximity: proximity to the ocean of the block group

10. median house value: median value of the houses of the block group

In this part of the coursework, you will implement a neural network architecture to infer the median

house value of a block group from the value of all other attributes.

You can find the skeleton code for this part in src/part2_house_value_regression.py. It defines a Regressor class that you will complete in the following questions. This file also defines two functions save_regressor and load_regressor, to save and load a Regressor model using pickle functions. You can modify these functions if you need to but make sure to change both of them so they can work in tandem. In this part, you must use either the PyTorch neural network library or the mini-library that you developed in the first part. You can find an introductory tutorial for PyTorch in the library documentation (https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html).

You can find the data for this part of the coursework in the file src/housing.csv. Using these data will be a bit more complicated than for the first part: they contain both numerical and textual entries, but also missing values in some of the columns. Thus, to read this csv file into python3, we use the function read_csv from Pandas library, as it allows to handle multi-datatype. Pandas is a widely-used Python library for data-sciences. It offers a nice interface to handle and visualise large data tables. You can find a quick introduction to Pandas here: https://pandas.pydata.org/pandas-docs/stable/ user_guide/10min.html. In the following steps, we will suggest some functions that may be useful for preprocessing the data. You are also free to use other libraries, such as SciPy, for preprocessing or implement your own preprocessor code. Note that the data arguments given during LabTS tests for Part 2 will be in the format of Pandas DataFrame objects.

The evaluation for this part of the coursework will be based on both your PDF report and LabTS test results. Make sure the report includes: 1) a description of your model, along with justification of your choices, 2) description of the evaluation setup, 3) information about the hyperparameter search you performed along with the resulting findings, 4) final evaluation of your best model. The maximum report length is 5 pages, including figures and tables. You are welcome to include additional details in an appendix, but the main report needs to contain the necessary information, and the markers are not required to take the appendix into account.

1This data was initially featured in the following paper: Pace, R. Kelley, and Ronald Barry. ”Sparse spatial autoregressions.” Statistics Probability Letters 33.3 (1997): 291-297.

</div>
</div>
<div class="layoutArea">
<div class="column">
7

</div>
</div>
</div>
<div class="page" title="Page 8">
<div class="layoutArea">
<div class="column">
Q2.1: Implement an architecture for regression

For this question, you will implement a neural network architecture to predict the median house value of a block group by completing the following methods of the Regressor class:

Preprocessor method: Implement the _preprocessor method that preprocesses the input and output of your model. It should do the following:

<ul>
<li>Handle Pandas Dataframe as input, as it will be used for the LabTS test, as explained previously.</li>
<li>Store some parameters used for the preprocessing to apply the same preprocessing method to all inputs of your model. It is important that any values necessary for data processing (e.g. normal- ising constants, mapping from categorical values to 1-hot vectors, etc.) are created based on the training data and the same values are applied during testing. You can use the boolean training parameter to determine whether the input is training data (and new dataset-wide preprocessing values should be calculated) or test/validation data (and existing values should be applied).</li>
<li>Handle the missing values in the data, for example setting them to a default value. You can use the Pandas function fillna to do it (https://pandas.pydata.org/pandas-docs/stable/ reference/api/pandas.DataFrame.fillna.html).</li>
<li>Handle the textual values in the data, encoding them using one-hot encoding. For this, you can use the Sklearn class LabelBinarizer (https://scikit-learn.org/stable/modules/generated/ sklearn.preprocessing.LabelBinarizer.html). Remember to store all the parameters you need to be able to re-apply your preprocessing again at test time.</li>
<li>Eventually normalise the numerical values to improve learning.
Constructor method: Implement the __init__ method to initialise your neural network model and all the attributes you need. This function takes as input the training data and apply the _preprocessor method to them to set the dimensions of the neural network model. You can add any input parameters you need to this method, but make sure to set them with a default value to avoid errors when testing on LabTS.

Model-training method: Implement the fit method to train your regressor model, performing the following steps number-of-epochs times:

• Perform forward pass though the model given the input.

• Compute the loss based on this forward pass.

• Perform backwards pass to compute gradients of loss with respect to parameters of the model. • Perform one step of gradient descent on the model parameters.

• You are free to implement any additional steps to improve learning (batch-learning, shuffling…).

Here is an example of how the Regressor class can be used to train a regressor model on the data:

<pre># Assuming x_train contains the data and y_train the corresponding targets
</pre>
<pre>regressor = Regressor(x_train, nb_epoch = 10)
regressor.fit(x_train, y_train)
save_regressor(regressor)
</pre>
Feel free to define any supplementary methods or classes you may find necessary. Detail your choice of architecture and the methodology you used in your report.
</li>
</ul>
</div>
</div>
<div class="layoutArea">
<div class="column">
8

</div>
</div>
</div>
<div class="page" title="Page 9">
<div class="layoutArea">
<div class="column">
Q2.2: Evaluate your architecture

The aim of this question is to propose an evaluation of your model by completing the following method:

Prediction method: Implement the predict method to predict the output corresponding to a given input using your model. Try to minimise the number of Python loops you use in this method using Torch tensors’ properties.

Evaluation method: Implement the score method to print or return indicators on the performance of your regressor model. You are allowed to use utilities from libraries such as scikit-learn.

Detail your choice of evaluation and the methodology you used in your report, and propose an analysis of your results. You might want to illustrate your choice using graphs or tables.

Q2.3: Fine-tune your architecture

Using the tools you have developed so far, perform a hyperparameter search using a well thought out methodology that you will detail in your report. You are allowed to use utilities from libraries such as scikit-learn. Implement this search in the RegressorHyperParameterSearch function. Find and save your best performing model, using the provided save_regressor function to generate a model file which will be used to load your model on LabTS. Detail the methodology you used in your report. You might want to illustrate your choices with graphs or tables showing the impact of the hyperparameters.

Note that the public LabTS tests will report your model performance on the same training data that has been provided to you. It is not suggested to directly tune for high performance on this test, as that would result in overfitting to the training data. The final private tests will measure the performance of your saved model on a separate held-out evaluation set.

Deliverables

You will have to submit two items on CATE:

• The SHA1 corresponding to the commit on you gitlab repository (used with LabTS). In this repository, you should have completed and pushed:

<ul>
<li>– &nbsp;Completed version of part1_nn_lib.py.</li>
<li>– &nbsp;Completed version of part2_house_value_regression.py.</li>
<li>– &nbsp;Uploaded your best model as part2_model.pickle. This has to be saved by save_regressor and will be loaded by load_regressor.</li>
<li>– &nbsp;Uploaded a README.md explaining how to run your code.

• Report (PDF) for Part 2 with a maximum length of 5 pages, including figures and tables. The

report should include:

<ul>
<li>– &nbsp;A description of your model, along with justification of your choices.</li>
<li>– &nbsp;A description of the evaluation setup.</li>
<li>– &nbsp;Information about the hyperparameter search you performed along with the resulting find-
ings.
</li>
<li>– &nbsp;The final evaluation of your best model.</li>
</ul>
</li>
</ul>
</div>
</div>
<div class="layoutArea">
<div class="column">
9

</div>
</div>
</div>
<div class="page" title="Page 10">
<div class="layoutArea">
<div class="column">
Important! Make sure to evaluate on LabTS early, to ensure that your code runs there correctly. You will lose points on any tests where the code fails or does not run.

Also, make sure to have at least one working commit tested on LabTS a few days before the deadline. The LabTS queues for running tests will likely get long as the deadline approaches and your tests might not complete on time if you don’t submit them early enough.

If you create your part2_model.pickle file on a machine with incompatible versions of libraries, then you might experience errors on LabTS. We recommend creating the pickle file in one of the lab machines, using the environment that is given in the spec. If you want to work on your own machine, make sure your python library versions match the ones in the lab environment.

</div>
</div>
</div>
