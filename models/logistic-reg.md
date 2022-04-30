# all things logistic regression

not actually a regression in the traditional sense because output is piped through a logistic function so that the output is between 0 and 1 (and so acting as a binary classifier than a regressor)

[good preread](https://towardsdatascience.com/the-basics-logistic-regression-and-regularization-828b0d2d206c)
[good definitions preread](https://www.sciencedirect.com/topics/computer-science/logistic-regression#:~:text=Logistic%20regression%20is%20a%20process,%2Fno%2C%20and%20so%20on.)

## thought exercise

suppose you have model A and model B both trained on feature 1 and feature 2 where both models are logistic regression models
suppose now model A was regularized while model B was not regularized
in addition, the data for model B had a shift, where all of feature 1 was multiplied by 100 (and feature 2 was left alone)
would the resulting two models be the same?

depends, only if feature 1 has low predictive power
since model B is not regularized, the coefficients are left to their own and can be pretty high. the key is that since feature 1 is scaled by 100, the coefficient of feature 1 can be scaled down by 100

since model A is regularized, both coefficients will see some decrease, with the less predictive power coefficient seeing more decrease. the difference here is that the coeffcient of feature 1 is decreased only so much to keep the overall error (w/ regularization) low. it may not be decreased by a scale of 100. also feature 2's coefficient is changed for model A but not for model B.

## misc notes

normalizing data puts data between 0 and 1, to help the logistic regression classifier learn the weights better

The primary difference between linear regression and logistic regression is that logistic regression's range is bounded between 0 and 1. In addition, as opposed to linear regression, logistic regression does not require a linear relationship between inputs and output variables. This is due to applying a nonlinear log transformation to the odds ratio

for logistic regression, data does not strictly need to be linearly separable but achieves very good performance with linearly separable classes

## regularization

L1 - Lasso (adds absolute magnitude of coefficients to error)
L2 - Ridge (adds squared magnitude of coefficients to error)

L1 or Lasso shrinks coefficients to zero, allowing for feature selection through regularization
L2 just reduces but I think is smoother since when less than one, squaring actually reduces the penalty eg. 0.5 * 0.5 = 0.25 amount of penalty
