# COURSE 2

## WEEK 1  

---  

**Train/Dev/Test Sets**  
Train set -> used for training the model
Dev set -> used for development. to tune the hyperparameters
Test set -> test the final model

general proportion 60/20/20
in practice, 99+/0.5/0.5 is much preferred
test set is optional  

**Bias/Variance**  
high bias -> model underfits. the model is trained on less details. both train set and dev set errors are high  
reduce high bias by making a larger nn.  

high variance -> model overfits. the model is trained on more details. dev set error is much higher than train set  
reduce high variance by having a larger dataset, if not possible use regularisation.

**Regularisation**  

Regularization is a technique used in machine learning to prevent overfitting and improve the generalization of a model. When a model is overfitting, it means that it is too complex and has learned to fit the training data too closely, resulting in poor performance on new, unseen data.

Regularization helps to address this issue by adding a penalty term to the loss function during training. This penalty term discourages the model from assigning too much importance to certain features or parameters, making the model more generalized and less prone to overfitting.

There are different types of regularization techniques, but the most common one is called L2 regularization or weight decay. In L2 regularization, the penalty term is proportional to the square of the weights of the model. By adding this term to the loss function, the model is encouraged to keep the weights small, which helps to prevent overfitting.

Regularization is an important concept in machine learning and is widely used in various algorithms, including neural networks. It is often combined with other techniques such as hyperparameter tuning and optimization to improve the performance of models.






