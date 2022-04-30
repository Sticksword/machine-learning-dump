# ML step by step

## initial data explorations

* [extensive guide to exploratory data analysis](https://towardsdatascience.com/an-extensive-guide-to-exploratory-data-analysis-ddd99a03199e)
* data profiling using `.shape` and `.describe()` and `.head()` and `.columns`
* more data profiling
  * number of uniq values for each variable: `.nunique(axis=0)`
  * unique values for a column: `df.some_column.unique()`
    * remove non-unique columns
* remove unwanted columns: `df_cleaned = df_cleaned.copy().drop(['url','image_url','city_url'], axis=1)`
* visualize data with matplotlib
* for features that are skewed, we can use a logarithmic transformer to make them as normally distributed as possible
* standardize/normalize data
  * normalized dataset will have values that range between 0 and 1
  * standardized dataset will have a mean of 0 and standard deviation of 1, but there is no specific upper or lower bound for the maximum and minimum values.
  * don't need to normalize for tree based models
    * main reason for normalization for error based algorithms such as linear, logistic regression, neural networks is faster convergence to the global minimum due to the better initialization of weights. Information based algorithms (Decision Trees, Random Forests) and probability based algorithms (Naive Bayes, Bayesian Networks) don't require normalization
* removing outliers:

  ``` python
  df_cleaned = df_cleaned[df_cleaned['price'].between(999.99, 99999.00)]
  df_cleaned = df_cleaned[df_cleaned['year'] > 1990]
  df_cleaned = df_cleaned[df_cleaned['odometer'] < 899999.00]df_cleaned.describe().apply(lambda s: s.apply(lambda x: format(x, 'f')))
  ```

### data distributions

The features in the dataset should conform to the statistical assumptions of the models.
ie. Many models implemented in Sklearn might perform poorly if the numeric features do not more or less follow a standard Gaussian (normal) distribution. Except for tree-based models, the objective function of Sklearn algorithms assumes the features follow a normal distribution.

So we should standardize (0 mean and variance of 1) and also log transform skewed features

* handling null values
  * count null for columns: `df.isna().sum()`
  * Delete rows with missing data: `df_cleaned = df_cleaned.dropna(axis=0)`
  * Mean/Median/Mode imputation
    * use median when there are a number of outliers that positively or negatively skew the data.
    * will decrease variance of the feature
  * randomly sample as imputation
  * Assigning a unique value
  * Predicting the missing values
  * Using an algorithm which supports missing values, like random forests
  * [good example comparing how you would imput using mean, median, most frequent, and a constant value](https://machinelearningmastery.com/statistical-imputation-for-missing-values-in-machine-learning/)
  * Mean imputation is generally bad practice because it doesn’t take into account feature correlation. For example, imagine we have a table showing age and fitness score and imagine that an eighty-year-old has a missing fitness score. If we took the average fitness score from an age range of 15 to 80, then the eighty-year-old will appear to have a much higher fitness score that he actually should.
  * Second, mean imputation reduces the variance of the data and increases bias in our data. This leads to a less accurate model and a narrower confidence interval due to a smaller variance.
  * for categorical features, can use most frequent imputation - do not use when a lot of missing values b/c it will distort the relationship of the most frequent categories with dependent variables
    * we can add another variable to capture that it was missing, but then it's adding more variables
* removing duplicates

## visualizing the data & looking for relationships

### histograms

``` python
import matplotlib.pyplot as plt
my_data.hist(bins=50, figsize=(20,15))
plt.show()
```

``` python
df_cleaned['year'].plot(kind='hist', bins=20, figsize=(12, 6), facecolor='grey', edgecolor='black')
```

### boxplots

``` python
### all dots outside of boxplot are outliers
df_cleaned.boxplot('price')
```

### scatterplots

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

# scatterplots for all your variable pairs
sns.pairplot(my_data)
```

### View correlation of variables via `.corr()`

``` python
corr_matrix = my_data.corr()
corr_matrix["attribute_a"].sort_values(ascending=False)

sns.heatmap(corr_matrix, xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
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

* Logistic Regression: basic linear classifier (good to baseline), need linearly separable data, good explainability
* Random Forest: ensemble bagging classifier, also good explainability
* K-Nearest Neighbors: instance based classifier
* Support Vector Machines: maximum margin classifier
* Gaussian Naive Bayes: probabilistic classifier
* XGBoost: ensemble (extreme!) boosting classifier

## document similarity

[a good guide on state of the art document similarity algorithms](https://towardsdatascience.com/the-best-document-similarity-algorithm-in-2020-a-beginners-guide-a01b9ef8cf05)

## my full project examples

* [spam text classification exploration for Square Marketing](https://github.com/Sticksword/to-spam-or-not-to-spam/blob/master/README.md)
