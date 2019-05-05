# Final Blog Post

## Vision

- The big idea:  
Our big idea was to create a movie recommendation system that would use different attributes of users and movies to create the most appropriate movie recommendation for a given user.

- Progress: 
We did make measurable progress towards this idea. We ended up creating two recommendation systems that used two different algorithms, one of them used k-means and the other one used collaborative filtering from movie-to-movie. We visualize the recommendations given by each of the systems and we concluded that they do give appropriate recommendations to each user.

## Data
We used a dataset containing 1 million ratings from 6000 users on 4000 movies. The data was already cleaned and stored in a csv file. We found this data in the movielens website and it is publicly available to everyone.

### Relative to its size was there enough information contained within it? 
Yes, the one million ratings was enough information for us to appropriately train the models, as well as to see certain trends in the data when we visualized it.

### Were you able to find what you wanted in the data? 
We struggles a little to find certain trends in the data at the beginning. We first look at how certain factors such as gender and occupation affect the type of movies a person prefers, but we did not find any clear trends. However, when we looked at age, we found some interesting trends, such as the fact that people tend to like children movies less as they get older. According to how much of an influence each of the factors analyzed had on movie preferences, we give them certain weights when creating our models.

### How did you collect, clean, and integrate it?
As mentioned previously, the data was already cleaned and stored in a csv file. We did not have to collect it nor scrape it.

## Methodology
### What did you do with your data? 
We created two recommendations systems and then compared their results. The first model was implemented using a KNN algorithm. For this model we created a matrix in which each vector represents a movie, and each index in that vectors represents the rating that each of the users gave that movie. We then measured the distance between each of those movie vector and looked at the closest movie vectors.

### What techniques were used to pick apart the data? 
The data was in csv files, so it was not difficult to use python to read through the files and extract featured necessary to our analysis. After extracting the data from the three major files (u.data, u.item, and  u.user) we created one master array that bridged together the three files using the foreign keys. For example, each u.data row included a key for a user, which we used to grab the user data from u.user and append it to the data array, creating a master array for that data point.
### Did you use ML? Stats? 
Yes. As we already mentioned previously we used a KNN algorithm and we also used various statistical measures to compare the accuracy of each of the models, such as Mean Squared Error and Cosine Similarity

We used Mean Squared Error to measure the performance of both models in the test data. Cosine similarity was used to measure the similarity between each of the movie vectors in the KNN model.

### How did you visualize your data?
We used mostly stacked charts and a TSNE plot to visualize our data. We used the stacked charts to visualize the raw data, and the TSNE plot to visualize the results for our models.


### Visualizations:


![](https://i.imgur.com/v3zZZPa.png)

![](https://i.imgur.com/eXPZQ9w.png)

![](https://i.imgur.com/X195dXO.png)

### Results
We analyzed the data in order to find some correlation between some of user’s characteristics and their movie preferences. We first looked at occupation and gender, but we did not find any interesting results. The preferences of movies across all occupations and genders was very even. However, we did find some interesting correlations between age and movie preferences. We found that people tend to like children movies less as they get older, as shown in the second visualization below. Also, we found the opposite correlation between age and drama movies, people tend to like drama movies more as they get older.
Using this information, we created a movie recommendation system that looks for similarity in movies, based on reviews by users, and returns the top 5 most similarly reviewed movies. In doing so, we transpose the movie representations into a 2D plane which makes it easy to visually display movie similarities.



# Previous Posts

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

![](http://5b0988e595225.cdn.sohucs.com/images/20180611/75341ecc471f450e97dd4635f6b94071.jpeg)

Association rules can also be used for recommendation. Items that are frequently consumed together are connected with an edge in the graph. Mining rules is not very scalable. The APRIORI algorithm explores the state space of possible frequent itemsets and eliminates branches of the search space, that are not frequent. Frequent itemsets are used to generate rules and these rules generate recommendations.
This method is especially effective in visualization

![](https://cdn-images-1.medium.com/max/1600/0*xkxhJLeXbka8k1NN)S

### Evaluation

To validate the recommending algorithms, we will divide the users into training set, while is fully submitted to the engine, and testing set, which is only partially submitted and used to evaluate the engine. The expected recommendation will be computed given observed data using RMSE (root mean squared error)

## Next Steps

For our next step, we will implement the classifier and the collaborative filtering algorithms on the database that is already cleaned and try to evaluate the result against each other.
