## Intro

We will develop a movie recommendation system such as the ones used in Netflix, Hulu and Youtube, that we will then try to adapt to other purposes such as book recommendations or even music recommendation system. Given the large amounts of data available nowadays regarding the preferences and habits of users we are able to create systems that infer the likes and dislikes of people and determine what sort of movies/readings/music the users would be fond of. 

We will be gathering data from existing databases containing information about movies and users and from websites containing similar information. We will then analyze this data and create a model using different sorts of recommendation systems. Data that will not have been used at that point will be utilized to test the accuracy of the different algorithms, and we will then select the algorithm or combination of algorithms that return the most accurate recommendations.

## Use

We consider this to be very relevant nowadays given that the algorithm can not only be used for movies but also for books, music and even shopping, such as the algorithms used by amazon. In addition, the large amounts of available data allow this kind of algorithms achieve high accuracy.

## Data

We are using a data set that contains ratings from different users on different movies. In particular, we are utilizing the MovieLens 1M dataset that has 1 million ratings from 6000 users on 4000 movies. Additionally, we plan on crawling websites where people rate movies such as IMDB to scrape additional data that we can include in our database. We plan on using BeautifulSoup to clean our scraped data and store relevant information.

As of now, we have a database consisting of cleaned MovieLens 1M data. Our database consists of three tables: Movies, Users, Ratings. The table schemas are listed below.

**Movies** id | title | genre
**Users**	id | gender | age | occupation | zipcode
**Ratings** user_id | movie_id | rating | timestamp

## Recommendation Engine

### Algorithms

- Classifier: Using a classification system would entail using the characteristics we have on an individual user to determine whether they would like a movie or not. For example, given a users age and gender, a model could determine if the user would enjoy The Avengers.  In the base case this classifier would respond with a simple Boolean response of Yes/No,  but it could be extrapolated to respond with a percent likelihood of enjoying the movie (ex. 1-10). In this case, we could easily rank these from highest to lowest likelihood of enjoyment.
To do this classification, we would need to loop through all available movies and run our classifier for each one. This would prove to be incredibly time intensive compared to some other approaches.
This algorithm would be best suited for a Deep Learning approach as it only depends on the individual user, unlike others which relate users and base recommendations dynamically. We could feed the model user information and their rankings for a specific movie, resulting in a model that can learn to predict ratings for a movie based on the users characteristics. For this to work, we would need a unique model for each movie which could result in a large amount of storage being used up. Another approach would be to use linear regression with multiple variables, although this may be less accurate.

- Movie Similarity/Content based algorithms

- Collabrative Filtering

### Evaluation

## Next Step


## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/hsalhab/MovieEngine/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/hsalhab/MovieEngine/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
