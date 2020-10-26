# ML step by step

## initial data explorations

* data profiling using `.shape` and `.describe()` and `.head()`
* visualize data with matplotlib
* standardize/normalize data
* handling null values
  * Delete rows with missing data
  * Mean/Median/Mode imputation
    * use median when there are a number of outliers that positively or negatively skew the data.
  * Assigning a unique value
  * Predicting the missing values
  * Using an algorithm which supports missing values, like random forests
  * [good example comparing how you would imput using mean, median, most frequent, and a constant value](https://machinelearningmastery.com/statistical-imputation-for-missing-values-in-machine-learning/)
  * Mean imputation is generally bad practice because it doesn’t take into account feature correlation. For example, imagine we have a table showing age and fitness score and imagine that an eighty-year-old has a missing fitness score. If we took the average fitness score from an age range of 15 to 80, then the eighty-year-old will appear to have a much higher fitness score that he actually should.
  * Second, mean imputation reduces the variance of the data and increases bias in our data. This leads to a less accurate model and a narrower confidence interval due to a smaller variance.
* removing duplicates

``` python
import matplotlib.pyplot as plt
my_data.hist(bins=50, figsize=(20,15))
plt.show()
```

``` python
from pandas.plotting import scatter_matrix

attributes = ["attribute_a", "attribute_b", "attribute_c",
              "attribute_d"]
scatter_matrix(my_data[attributes], figsize=(12, 8))
```

``` python
my_data.plot(kind="scatter", x="attribute_a", y="attribute_b",
             alpha=0.1)
plt.axis([0, 16, 0, 550000])
```

View correlation of variables via `.corr()`

``` python
corr_matrix = my_data.corr()
corr_matrix["attribute_a"].sort_values(ascending=False)
```

## data considerations

### sampling

* Sampling bias: a biased sample caused by non-random sampling
* Under coverage bias: sampling too few observations
* Survivorship bias: error of overlooking observations that did not make it past a form of selection process.

There are many things that you can do to control and minimize bias. Two common things include randomization, where participants are assigned by chance, and random sampling, sampling in which each member has an equal probability of being chosen.

We can also employ stratefied sampling to respect the underlying distribution.

## model comparisons

### why random forests > SVMs

* Random forests allow you to determine the feature importance. SVM’s can’t do this.
* Random forests are much quicker and simpler to build than an SVM.
* For multi-class classification problems, SVMs require a one-vs-rest method, which is less scalable and more memory intensive.

#### breakdown

* Logistic Regression: basic linear classifier (good to baseline)
* Random Forest: ensemble bagging classifier
* K-Nearest Neighbors: instance based classifier
* Support Vector Machines: maximum margin classifier
* Gaussian Naive Bayes: probabilistic classifier
* XGBoost: ensemble (extreme!) boosting classifier

## document similarity

[a good guide on state of the art document similarity algorithms](https://towardsdatascience.com/the-best-document-similarity-algorithm-in-2020-a-beginners-guide-a01b9ef8cf05)

## my full project examples

* [spam text classification exploration for Square Marketing](https://github.com/Sticksword/to-spam-or-not-to-spam/blob/master/README.md)