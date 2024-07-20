# COURSE 1 -
## WEEK 1-
**Introduction to deep learning and neural network**

* A single neuron takes some input and gives an output and many neurons stacked together form a neural network

* Supervised learning – uses neural network where input x and output y are both given and model is trained to predict y for some x

* Different neural networks –
Standard NN, Convolutional NN ( image processing ), Recurrent NN ( sequential data like time series, audio,etc ), Hybrid NN

* Types of data-
Structured data- each feature has a well defined meaning with labels 
Unstructured data – raw audio, images not well defined structure

* Performance of a NN increase with increase in amount of data. Scaling of data and parameters has improved deep learning by improving computation and size of NN

## WEEK 2-
**Basics of neural network programming**

We will learn :
-forward and backward propagation
-eliminating the use of for loops for neural networks

### LOGISTIC REGRESSION -

_NOTATIONS -_

* (x_i,y_i) – i ranges from 1 to m
* x is called feature y is the label
* can be a vector with n_x parameters
* Logistic regression is used for classification ie y can be either 0 or 1

* m -> no. of training examples  
* X is a matrix of size (n_x,m)
* Y is a matrix of shape (1,m)
* y^ is predicted output = P(y=1|x)

_SIGMOID FUNCTION –_

* We need sigmoid to give a value between 0 and 1 since z by itself can be any value
![sigmoid](Notes\Images\sigmoid.png)
 
_LOSS FUNCTION – L_

* Measures how good y^ is as compared to y 
* So this ensures loss is minimum for both y=0 and y=1 when y^ is a good prediction ( ie close to y)
![loss](Notes\Images\loss.png)

_COST FUNCTION – J_

* Loss function was for single training example , cost is for entire training set
* It is a convex function with only 1 global minima 
![cost](Notes\Images\cost.png)

_GRADIENT DESCENT -_

* It helps us find w and b that minimize the cost function J ( ie give y^ almost close to y) through backpropagation
* We update w and b in every iteration using below function and determine cost after every iteration until it becomes say minimum
![gd](Notes\Images\gradient_descent.png)

_COMPUTATION GRAPHS –_

* Computation involves -
Forward propagation to compute output of a neural network followed by backpropogation which changes w and b using gradient descent

_VECTORIZATION –_

* Using np.dot to multiply two vectors instead of using for loop reduces computation time and is useful for large datasets.
* Both GPU and CPU have parallel instructions – SIMD which works in case of dot product 
![v](Notes\Images\vectorization.png)

_BROADCASTING –_

* Using this we can resize vectors/scalars to give a shape appropriate for performing arithmetic operations with other matrices. 
* For (m,n) + (1,n) -> the (1,n) resizes to (m,n) 
* For (m,n) + k -> k changes to (m,1) with all m values as k 
