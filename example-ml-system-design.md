# ML System Design: Game Recommendation

Let's say we work at Steam and are tasked with building out the recommendation system for the Steam store homepage.

The scale of the platform is around 100K+ titles with 100M+ MAUs.

Fun twist, how does this change if we work at Roblox, where there are millions of games/experiences and 3x the MAU.

Identify the top line metrics. Are we aiming to move MAUs, DAUs, total time played, something else?

Are we building from scratch or are we improving an existing system?

Say there is a rule based system in place and we need to improve upon this.

At the scale of 100K+ titles, we can come up with a two stage process for creating game recommendations. Latency is a constraint as players don't want to wait for their store homepage to load. Having two stages allows for us to break apart the recommendation process and employ different strategies to ensure we stay under the latency limit. Some strategies and algorithms are more compute intensive and so we may not want to just throw 100K+ games through this. Rather with our two stage approach, the first stage will sift through and go from 100K+ titles to 1,000 candidates that the user may like. Our second stage then uses more compute intensive methods to go from 1,000 to the top 100. We can then showcase those top 100.

For stage 1, we can employ either collaborative filtering methods (eg. matrix factorization) or we can employ content based methods (eg. two tower neural network).

CF methods pros:
* more efficient and simple
* no domain knowledge is needed as everything is based off user interactions rather than custom feature engineering

CF methods cons:
* struggles with cold start problem
* interaction data can be sparse, making it harder for niche items/interests

Content based methods pros:
* better at sifting and recommending new items since it's content based
* can capture unique interests and make things more deeply personalized

Content based methods cons:
* more effort, more domain knowledge needed to model users' interests (and tune such model)

Both are meant to create a fast and scalable means to reduce the candidate space to a more personalized subset. Using each is not mutually exclusive either, and so as the system continues to grow, we can evaluate having multiple of these "candidate generators".

Each of these stage 1 approaches will need logging setup to capture the implicit labels necessary to train such models.

For stage 2, the model architecture changes. We are no longer tied to keeping separate towers where the user tower only learns about the user and the item tower only learns about the tower. This means the model could increase significantly the parameter size and also time needed to process the underlying user and item during serving. Additionally we can employ longer sequence features such as longer user:item features that we normally may not choose to use in our stage 1 two tower model. So with all this in mind, it would be super resource intensive to throw millions or even billions of items through this model.

Once our system matures, we can explore additional methods to showcase our titles through dedicated groupings and varied UI approaches. One such approach would be to employ "widgets" or "rows" or "carousels". With this we are now considering not only item ranking but also row ranking. This would fall under the category of whole page construction as we now are trying to optimize not just the best items to showcase but also how to best present those items.


Serving
- use docker
- use some kind of model registry
- package model and other needed files, deploy onto a platform that will handle spinning up the machine to host your new model
- ideally some autoscaling config
- feature store vs bundled csv
- log the model inference results

Monitoring & Observability
- grafana for qps/error rates
- feature monitoring, distribution drift
- model performance monitoring, CTR, purchase rate, etc.
