**Overview**

This project involves sentiment analysis on textual data using two machine learning algorithms: Logistic Regression and Multinomial Naive Bayes. The dataset contains tweets labeled with sentiments, and the goal is to predict the sentiment of tweets in the test dataset. The project demonstrates various steps of text preprocessing, feature extraction, model training, and evaluation.

**Dataset**

Train Dataset: train.csv

Test Dataset: test.csv

Both datasets contain tweets with their corresponding sentiment labels. The datasets are loaded and explored for missing values and basic statistics.

**Libraries Used**

nltk: Natural Language Toolkit for text processing

pandas: Data manipulation and analysis

matplotlib & seaborn: Data visualization

re: Regular expressions for text processing

sklearn: Machine learning library

wordcloud: Visualization of text data

numpy: Numerical computations

**Data Preprocessing**

Text Cleaning:

Removal of URLs, hashtags, mentions, and non-alphabet characters.

Conversion of text to lowercase.

Removal of extra spaces.

Stemming:

Reduction of words to their root forms using PorterStemmer.

Stopword Removal:

Removal of common English stopwords using nltk's stopword list.

TF-IDF Vectorization:

Conversion of text data into TF-IDF features with a maximum of 100,000 features.

**Models**

1-Logistic Regression:

Implementation using sklearn.linear_model.LogisticRegression.

Trained with L2 penalty and a maximum of 500 iterations.

2-Multinomial Naive Bayes:

Implementation using sklearn.naive_bayes.MultinomialNB.

**Model Evaluation**
Accuracy, confusion matrix, and classification report are used to evaluate the performance of the models.

Visualizations of confusion matrices and accuracy comparison between the models are provided.

**Visualizations**
Distribution of sentiments in the training dataset.

Confusion matrices for both models.

Bar plot comparing the accuracy of the two models.

**How to Run**

1-Clone the repository:

git clone <repository_url>

2-Navigate to the project directory:

cd sentiment-analysis-project

Ensure that the requirements.txt file is in the project folder.

3-Install the required packages:

pip install -r requirements.txt

4-Ensure that train.csv and test.csv are loaded into the project directory before executing the code.

Run the notebook or script to preprocess data, train models, and visualize results.

**Conclusion**

This project demonstrates the process of text preprocessing, feature extraction, and model training for sentiment analysis. The Logistic Regression model performed better than the Multinomial Naive Bayes model in terms of accuracy.
