# duolingo system practice question

Duolingo is a platform for language learning. When a student is learning a new language, Duolingo wants to recommend increasingly difficult stories to read.

* How would you measure the difficulty level of a story?
* Given a story, how would you edit it to make it easier or more difficult?

## measuring difficulty

difficulty of a story could consist of how long the story is, the average size of words (assuming Latin based language), the tf-idf score of the words (given we compute these scores across some corpus of documents like the common crawl - was it wikipedia that was crawled?), number of "high school level/college level" words it contains, etc.

we can also manually label stories at the beginning as the training dataset. we can then build some ml model to predict the difficulty based on the characters/words/sentences in the story. this will allow us to roughly link the text characteristics of a story to a difficulty rating

## editing stories

for non-embedding signals, we can use feature importances (SHAP, etc) to determine how it impacts story difficulty. we can then dial that feature back to make the story less difficult. f
