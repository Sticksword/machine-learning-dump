# all things ml system

[good preread](http://patrickhalina.com/posts/ml-systems-design-interview-guide/)

## example flow given by educative.io

1. problem statement
  clarify when given problem statement
  eg. design linkedin feed system
  "ok how do we want to balance content vs ads"

2. identify metrics
  how do we evaluate offline models
  eg. AUC for roc and pr curves

3. identify requirements
  pin down training requirements such as data collection, feature engineering, feature selection, and loss function
  eg. "how do we train a model to handle class imbalance?"
  can also pin down inference requirements such as high availability and low latency

4. train and evaluate models
  feature engineering, feature selection, models

5. design high level system
  think through high level system components and how data flows between then

6. scale the design
  understand system bottlenecks and how to address them

## example thought process by patrick halina (staff ml eng @ pinterest)(see preread)

likely will get asked to design some kind of recommender system

1. product requirements clarification
  real time or batch? apply to all users or a specific segment? what does "show the most recent content" mean? 5 minutes or 24 hours?
2. high level design
  eg. two stage architecture for recommender systems -> 1. candidate generation (output hundreds) and 2. ranking (output 10ish)
3. data brainstorming and features
  data sources, high level features, feature represenation (embeddings often play a big role), feature selection
4. model development
which model type are we thinking:

- Binary classifiers
- Multi-classifiers
- Regressions

For our recommender example, _the ranking component can be built with an ML model_. We can _rank the candidates by their predicted outcome for the user_. For example, maybe based on our initial discussions, we’re trying to increase engagement by showing posts that increase user interactions with the posts. There’s lots of ways to do this:

- A binary classifier that predicts whether the user will interact with the given post
- A multi classifier with predictions for commenting, liking or hiding a post
- A regression that predicts the number of interactions a user will make with the post

For offline training, how will we use the data? For example, if we’re training a binary classifier to predict whether a user will ‘like’ a post, there’s a lot more posts in the world that the user didn’t like than liked! Should we only train the model using posts that the user observed in their feed which they didn’t like? What if they ended up liking a post later on, do we only label data based on their first impression? How do we deal with data imbalances which are very common in recommender systems?

For classifiers you should discuss what’s more important, precision or recall, especially considering their effect on users. How does it affect the user experience to have a ‘false positive’ versus missing the ‘right’ answer?
