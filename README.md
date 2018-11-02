# Sentiment-Analysis-on-Social-Media-Messages

Files:
All these files are placed in a folder named data.
1.	train.dat -> Training Data
2.	test.dat -> Test Data
3.	shortforms.xlsx -> Map of short forms
Command to run the code:
python3 code.py
Steps:
1.	Import all the necessary libraries.
Libraries used  sklearn, numpy, pandas and nltk.

Pandas --- used for loading the data from files into data frame.
Numpy --- used to save the predicted output to .dat file
Sklearn --- used for text preprocessing i.e. TF-IDF Vectorizer,
	     Model Generation (SVM, Logistic Regression and KNN)
NLTK --- Used for preprocessing of text: 
Tokenization, 
Stemming and 
Removal of stop words.

2.	Load the Training Data (file path = “data/train.dat”)
Rows: 31000
Columns: 2 i.e. Contains two columns: Text(X) and Sentiment(Y)

3.	Text Preprocessing:
•	The text data is unprocessed and raw data it must be cleaned and processed so that it can used as input for the model to learn. 
•	Preprocessing is composed of several smaller steps.
a.	Expand all the short forms present in the data using map called contractions.
It contains entries like:
Key: isn’t 
Value: is not.
		So the line ‘Bob isn’t home’ is modified as ‘Bob is not home’.
b.	Count all the stop words. Stop words are taken from nltk library filtering a few of them.
Bob is not home. -> contains 1 stop word is
c.	Convert text to lowercase, free of special characters (punctuation) and stop words
Bob is not home. -> bob not home
d.	Replace all the shortcuts of messaging slang with actual words.
For this step, we use shortforms.xlsx (file path = “data/ shortforms.xlsx”). which is the data taken from this URL.
E.g.  It contains all the social media words for example:
	b4 -> before
e.	Also removed the multiple occurrences of a character if the count is greater than two. 
E.g. woooooow  -> woow
f.	Repeat step c after replacing shortcuts.
g.	Stem the words to its root word using Porter Stemmer from nltk library. 
Tried different stemmers like Snowball Stemmer and Lemmatizer. 
Got best results with Porter Stemmer.
4.	Use TF-IDF Vectorization to convert the text into term frequency, inverse document frequency vectors. (i.e. convert doc to matrix). 
5.	Employed the use of k-grams at this step (1-3 k-grams).
6.	Divide the training data into train and validation sets with validation set as 20% of the train data.
7.	Load the Test Data (file path = “data/test.dat”). Repeat same preprocessing steps on Test data.
8.	Classifier Model Selection (Tried different models and got best results with the below three models):
a.	Linear Support Vector Classification (Linear SVM Model) 
Hyper parameters: 
Loss = hinge, 
Maximum number of iterations =1000
b.	Logistic Regression Model
Hyper parameters: 
Maximum number of iterations =1000
c.	K nearest neighbors Model
Hyper parameters: 
Number of neighbors = 3
Distance metric = minkowski
9.	Fit the train data to all the models.
10.	Make predictions using these models on validation set.
11.	Comparing their F1-scores by making predictions on validation set using the above three models and chosen the model giving best F1-score i.e. Logistic Regression model for classification.
12.	Make the predictions on the test data.
13.	Save the predictions to output file (predictions.dat).
