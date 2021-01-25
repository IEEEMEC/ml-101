# Mathematical Concepts of Linear Regression

## Linear Regression
Let us go through the mathematical concepts of linear regression. The variable we are predicting
is called the criterion variable and is referred to as Y. The variable we are basing our predictions
on is called the predictor variable and is referred to as X.
For example, suppose we have a variable Y (say, the sales received by a company after
advertising) that we want to predict using a variable X (the money the company spent on TV
advertisements). We can model the relationship between the two as:

![alt text](https://github.com/allenabraham999/SiG/blob/main/images-2/Img1.jpg)

β0 and β1 are the intercept and the slope respectively, and combined are the
coefficients or parameters of this model. We use the [training data](https://www.techopedia.com/definition/33181/training-data#:~:text=The%20training%20data%20is%20an,called%20validation%20and%20testing%20sets.) to create estimates of these
values( here, 0 and 1
), hence we don't use the normal '=' sign, and then we can use them to
predict company sales for potential spending on advertisements. You can see how that might be
useful.

![alt text](https://github.com/allenabraham999/SiG/blob/main/images-2/Img2.jpg)
Now, how do we use the training data to find the values for these unknowns? Let's say we have
an n number of data entries of the form (xi
, yi
). Now if we plot these values out on a graph, we'll
want the slope and intercept of the line that's closest to these points. There are many ways to
calculate this closeness, but the most common one in this scenario is minimizing the
least-squares. A least-square method is a form of mathematical regression analysis used to
determine the line of best fit for a set of data, providing a visual demonstration of the relationship
between the data points. Each point of data represents the relationship between a known
independent variable and an unknown dependent variable.
Let ŷi be the predicted value of Y for a data point xi among these data points (the real value being
yi
). The error ei can be seen as

![alt text](https://github.com/allenabraham999/SiG/blob/main/images-2/Extra.jpg)

This value is what we call the ith residual the residue left in the predicted value relative to the
actual value. Now you might think the obvious solution is to minimize these residuals. But how
would you do that? It's not like we can generate every possible line to see which gives us the
lowest residuals. No, the solution lies in mathematics. So to easily calculate the lowest possible
residuals, we need some way to take a double derivative of the residuals. That's why we use
the residual sum of squares (RSS), which is calculated as:

![alt text](https://github.com/allenabraham999/SiG/blob/main/images-2/Img3.jpg)

Now, we can simply try to find the minimum value of this RSS. The equations for β0 and β1

![alt text](https://github.com/allenabraham999/SiG/blob/main/images-2/Img4.jpg)

where ȳ and x̄ are the mean values for x and y:

![alt text](https://github.com/allenabraham999/SiG/blob/main/images-2/Img5.jpg)

And as a bonus, this is what the graph for that would look like. The red dot is the lowest point in
that graph and thus, the lowest value of RSS and β0 and β1 are simply the x and y coordinates of
that point. This method, as it happens, is how optimization for most AI models are performed -
mathematically finding the lowest point on a graph measuring inaccuracy.

## Multiple Linear Regression.

Multiple linear regression (MLR), also known simply as multiple regression, is a statistical
technique that uses several explanatory variables to predict the outcome of a response variable.
In essence, multiple regression is the extension of ordinary least-squares (OLS) regression that
involves more than one explanatory variable. Formula and calculation of multiple linear regression
is given by,

![alt text](https://github.com/allenabraham999/SiG/blob/main/images-2/Img6.jpg)

Maybe you're a large company. You have the budget to spend on more than TV advertisements.
Maybe you advertise in the newspaper and on billboards as well. How would you then find out the
optimal amount to spend?

![alt text](https://github.com/allenabraham999/SiG/blob/main/images-2/Img7.jpg)

Here, βi stands for the relationship between Xi and Y, independent of all other variables. In other
words, βi
is the change in Y for one unit of change in Xi, everything else remaining the same. Now
I'll tell you something weird that happens, and you try and see if you can predict why it happens.
When we try to plot the graph between sales and money spent on TV of this model, we find that
somehow, β1
is close to 0; or more simply, sales do not change when we spend more on TV
advertising. Can you guess why?

Well, it turns out that whenever your marketing team spends more money on TV, they also spend
more on newspapers and billboards. In the simple linear regression, you aren't just calculating
how much sales will increase when TV advertising increases - you're calculating how much sales
will increase when every situation where TV advertising could increase occurs. Now, in the
multiple linear regression, where you take the change in the TV spending while newspaper and
billboard budgets remain the same, you see that it's not very relevant at all.


### Estimating Accuracy
The one thing true for all machine learning methods, whether it is a decision tree or deeplearning: you want to know how well your model will perform.
You do this by measuring its accuracy

Why? First of all, because measuring a model’s accuracy can guide you to select the
best-performing algorithm for it and fine-tune its parameters so that your model becomes more
accurate.
But most importantly, you will need to know how well the model performs before you use it in
production.
If your application requires the model to be correct for more than 90% of all predictions but it only
delivers correct predictions 80% of the time, you might not want the model to go into production at
all.
So how can we calculate the accuracy of a model? The basic idea is that you can train a
predictive model on a given dataset and then use that underlying function on data points where
you already know the value for y.

## Training error vs test error
There are two important concepts used in machine learning: the training error and the test error.
● Training Error: We get this by calculating the classification error of a model on the same
data the model was trained on (just like the example above).
● Test Error: We get this by using two completely disjoint datasets: one to train the model
and the other to calculate the classification error. Both datasets need to have values for y.
The first dataset is called training data and the second, test data.
As we already saw, the training error rate is quite different from the testing error rate. The test
error rate can be calculated if we have a dedicated testing set available, but that's not always the
case. In that scenario, assessing a model using its training error rate can lead to overconfidence in
its accuracy. To finish off Part III, we'll look at some of the methods for estimating the test error
rate by holding out on some of the training data.
The obvious solution is to split the training dataset into two parts and use one for testing, a
solution is known as the validation set approach. But the problem that arises is that the model's
test accuracy will depend heavily on which data is in the first part, the part it's trained on. And
since models are inclined to perform better with more training, this might be a situation where if we
train it on only a part of the data, it'll overestimate the test error.
What if we could get more training while also keeping our accurate tests? That's the idea behind
Leave-one-out cross-validation (That's a lot of words, so let's just call it LOOCV). In it, we train the
model on the entire training data, excluding just one observation, which we'll use for testing. And
since the test error (the mean square error MSE = (yi – ŷi)2) here will be very variable because
we're using just one observation instead of averaging over an entire testing dataset, we repeat this
for every single observation. In other words, we retrain the model using a different observation as
test data each time, until every observation has been used. The test error could be the average of
the errors from each training.

![alt text](https://github.com/allenabraham999/SiG/blob/main/images-2/Img8.jpg)

But this can be expensive, training the model so many times.
What if we took several observations from the dataset each time, for testing? It'd look much like
the validation set approach in the first iteration, but with the added benefit of retraining so the
model can fully utilize the training data. We'd divide the entire dataset into k parts, and use one
part for testing each time. This is called k-Fold Cross-Validation. LOOCV is just a special case of
this where k = 1.

![alt text](https://github.com/allenabraham999/SiG/blob/main/images-2/img9.jpg)

As you can see, the k-fold method performs almost exactly as well as LOOCV (the figures
mentioned in the descriptions aren't relevant, they're just normal datasets), for far less computing
expense.
But computational issues aside, it turns out that k-fold cross-validation more often gives accurate
estimates of the true test error rate than LOOCV (This can't be seen very well in the above
diagram, but it happens quite a lot). This relates to the [bias-variance tradeoff](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229).
As we stated above, the validation set approach will overestimate the test error rate, while
LOOCV, where every training set is n - 1 in length, will give a more accurate (or less biased)
estimate. So k-fold cross-validation would have an intermediate level of bias, and from this angle,
LOOCV should be preferred to it.
But it turns out that LOOCV has a higher variance than k-fold CV (when k < n). This occurs
because what LOOCV effectively does is average the n models it creates, which are all very
similar. After all, they're trained on almost the same data. In k-fold CV, the models are still
somewhat similar, but less so because the models have a more varied training dataset. The
validation set approach would have even lower variance in that regard.
Thus to summarize, we can say the problem is that the choice of k involves a bias-variance
tradeoff (k = 1 is LOOCV, k = n/2 or the like would be the validation set approach), and so
typically, one uses this method taking k = 5 or k = 10, as there is empirical proof that they yield test
error estimates that suffer neither from extremely high bias or excessive variance.

### Choosing Models

Model specification is the process of determining which independent variables to include and
exclude from a regression equation. How do you choose the best regression model? The world is
complicated, and trying to explain it with a small sample doesn’t help.
The need for model selection often begins when a researcher wants to mathematically define the
relationship between independent variables and the dependent variable. Typically, investigators
measure many variables but include only some in the model. Analysts try to exclude independent
variables that are not related and include only those that have an actual relationship with the
dependent variable. During the specification process, the analysts typically try different
combinations of variables and various forms of the model. For example, they can try different
terms that explain interactions between variables and curvature in the data.
In simple linear regression, we predict scores on one variable from the scores on a second
variable. The variable we are predicting is called the criterion variable and is referred to as Y. The
variable we are basing our predictions on is called the predictor variable and is referred to as X.
For example, in a plant growth study, the predictors might be the amount of fertilizer applied, the
soil moisture, and the amount of sunlight.
If the relationship between the predictors (factors) and the output is linear, you use linear
regression. If the relationship is non-linear and more quadratic, logistic regression works well. If it's
very non-linear and complex, trees might be a better option.
Trees are popular for more reasons than their predictive power - they're easy to understand by
looking at them, and that's always nice. Unfortunately, they aren't quite as accurate as some of the
other regression techniques and even a small change in the training data can result in a large
change in the final tree.

![alt text](https://github.com/allenabraham999/SiG/blob/main/images-2/Img10.jpg)
