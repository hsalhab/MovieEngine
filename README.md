## Intro

We will develop a movie recommendation system such as the ones used in Netflix, Hulu and Youtube, that we will then try to adapt to other purposes such as book recommendations or even music recommendation system. Given the large amounts of data available nowadays regarding the preferences and habits of users we are able to create systems that infer the likes and dislikes of people and determine what sort of movies/readings/music the users would be fond of.

We will be gathering data from existing databases containing information about movies and users and from websites containing similar information. We will then analyze this data and create a model using different sorts of recommendation systems. Data that will not have been used at that point will be utilized to test the accuracy of the different algorithms, and we will then select the algorithm or combination of algorithms that return the most accurate recommendations.

## Use

We consider this to be very relevant nowadays given that the algorithm can not only be used for movies but also for books, music and even shopping, such as the algorithms used by amazon. In addition, the large amounts of available data allow this kind of algorithms achieve high accuracy.

## Data

We are using a data set that contains ratings from different users on different movies. In particular, we are utilizing the MovieLens 1M dataset that has 1 million ratings from 6000 users on 4000 movies. Additionally, we plan on crawling websites where people rate movies such as IMDB to scrape additional data that we can include in our database. We plan on using BeautifulSoup to clean our scraped data and store relevant information.

As of now, we have a database consisting of cleaned MovieLens 1M data. Our database consists of three tables: Movies, Users, Ratings. The table schemas are listed below.

**Movies** | id | title | genre
**Users**	| id | gender | age | occupation | zipcode
**Ratings** | user_id | movie_id | rating | timestamp

## Recommendation Engine

### Algorithms

- Classifier: Using a classification system would entail using the characteristics we have on an individual user to determine whether they would like a movie or not. For example, given a users age and gender, a model could determine if the user would enjoy The Avengers.  In the base case this classifier would respond with a simple Boolean response of Yes/No,  but it could be extrapolated to respond with a percent likelihood of enjoying the movie (ex. 1-10). In this case, we could easily rank these from highest to lowest likelihood of enjoyment.
To do this classification, we would need to loop through all available movies and run our classifier for each one. This would prove to be incredibly time intensive compared to some other approaches.
This algorithm would be best suited for a Deep Learning approach as it only depends on the individual user, unlike others which relate users and base recommendations dynamically. We could feed the model user information and their rankings for a specific movie, resulting in a model that can learn to predict ratings for a movie based on the users characteristics. For this to work, we would need a unique model for each movie which could result in a large amount of storage being used up. Another approach would be to use linear regression with multiple variables, although this may be less accurate.

- Movie Similarity/Content based algorithms

Content based methods are based on similarity of item attributes and collaborative methods calculate similarity from interactions of the single user.
Cosine similarity is used to compute the similarities between films. The method vectorizes the movie data (in terms of words in the title, or rating), and then apply the cosine-similarity function.

- Collabrative Filtering

Collaborative methods on the contrary enable users to discover new content dissimilar to items viewed in the past, using ratings of all users in the database. A couple of the most-used collaborative methods are listed below:

K nearest neighbors
The simplest algorithm computes cosine or correlation similarity of rows (users) or columns (items) and recommends items that k . Here is the pseudocode for KNN:

![](http://p1.qhimgs4.com/t0109802bc8cecae70d.jpg)

Matrix factorization
Matrix factorization based methods attempt to reduce dimensionality of the interaction matrix and approximate it by two or more small matrices with k latent components. By multiplying corresponding row and column you predict rating of item by user. Training error can be obtained by comparing non empty ratings to predicted ratings. One can also regularize training loss by adding a penalty term keeping values of latent vectors low.

Association rules can also be used for recommendation. Items that are frequently consumed together are connected with an edge in the graph. Mining rules is not very scalable. The APRIORI algorithm explores the state space of possible frequent itemsets and eliminates branches of the search space, that are not frequent. Frequent itemsets are used to generate rules and these rules generate recommendations.
This method is especially effective in visualization


### Evaluation

To validate the recommending algorithms, we will divide the users into training set, while is fully submitted to the engine, and testing set, which is only partially submitted and used to evaluate the engine. The expected recommendation will be computed given observed data using RMSE (root mean squared error)

## Next Steps

For our next step, we will implement the classifier and the collaborative filtering algorithms on the database that is already cleaned and try to evaluate the result against each other.
