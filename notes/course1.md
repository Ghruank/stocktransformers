# COURSE 1

## WEEK 1

week 1 covers neural network basics, introduction to deep learning, how a simple neural network works and how it can be made more efficient and complex by adding hidden layers

supervised learning-> where inputs and outputs are available and the model is trained on that

structured data-> data where there is a structure, like X and Y columns, which are labelled
unstructured data-> data where there is no structure, like images or speech or here, time series

simple NNs are used for structured data, CNNs are used for images and stuff
RNNs are used for sequential data like speech and time series

performance of a NN increases with-> 1) increase in amount of data and 2) making a larger NN
depends on data, computation and the algorithm used

## WEEK2

in week2, the following stuff is covered:
1. Techniques for processing a training set without using an explicit for loop
2. The concept of forward propagation and backward propagation in neural network computation
3. Introduction to logistic regression as an algorithm for binary classification
4. Representation of images in a computer and how to convert pixel intensity values into a feature vector
5. Notation and representation of training examples and labels
6. Introduction to matrix notation for input features and output labels

**Logistic Regression**  
Logistic regression is a statistical method for modeling the probability of a binary outcome based on one or more predictor variables. It uses the logistic function to model the probability of the default class.  

we have an x vector, we multiply it by w(t) (transpose of vector w which has same dimension of x) to make it a scalar quantity,add a scalar b to it, take the sigmoid of it for it to lie between 0 and 1, and this becomes the y^, which is thepredicted value of y. ofcourse since w and b are random, y^ will not be equal to y, and as such we find w and b through backpropogration, specifically done using gradient descent. the loss function and the cost function helps to find the error between y^ and y, and these function need to be minimised for y^ to be almost equal to y. these functions can be minimised using gradient descent, as they are related to dw and db, and when dw or db is 0, we can say that J(w,b), which is a convex 3d graph, has attained minimum. this w is updated as w = w - a.dw, where a is the learning rate.  

*Terminologies*  
Sigmoid(z) = 1/1+e^-z  
Loss (error) function = L(y^,y) = -[y log(y^) + (1-y) log(1-y^)]  
Cost function = J(w,b) = 1/m * summation[L(y^,y)]  

**Computation graph**  
this helps us to find the structure of the NN, using forward and backpropogation
forward propogation used to find the required equation
back propogation used to find the gradient descent and weights/biases

**Vectorisation**
used instead of for loops. for loops iterate through each of the data set and do the operation. using numpy we perform vectorisation where all of the operations are performed at once, reducing time required to process the data.
eg Z = np.dot(w,x) + b



