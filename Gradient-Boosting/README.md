# Gradient-Boosting

Gradient boosting is a machine learning technique that consistently improves the quality of predictions by combining weak models into a strong model.

The main idea is to train a new model that will correct the "errors" of the previous model, gradually improving the quality of the predictions at each iteration.

This is achieved by finding the gradient of the loss function (eg MSE or MAE) with respect to the predictions of the current ensemble of models and training the new model on those errors.

