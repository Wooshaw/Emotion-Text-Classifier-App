<center><h2>Emotion-Text-Classifier-App</h2></center>

<h2>Table of Contents </h2>

1. <a href="#introduction">Introduction</a> 
2. <a href="#project">Project Description</a> 
3. <a href="#todo">TODO</a> 
  
<h2 id="introduction">Introduction </h2>
<p>Sentiment analysis from digital text has became one of the most popular topics to study in the fields of natural langugae processing. It is especially important for many businesses to better understand their customer when performing brand management, social media marketing, or survey analysis at scale. With the use of the large data of tweets and Facebook post online, I can use machine learning model to produce a meaningful outcome for business application.<p>

<h2 id="project">Project Description </h2>

### Aim
This is a multi-class sentiment analysis project that classifies texts into eight different emotion categories using traditional machine learning and tranfer learning models using BERT.

**Requirements:** 
- Python 3.xx
- Installation of libraries and packages included in the requirements.txt
- Love for machine learning

### Results

#### Data Cleaning 
<img width="380" alt="Screen Shot 2023-01-25 at 12 17 46 AM" src="https://user-images.githubusercontent.com/102776898/214494068-00ae1419-f718-4595-a771-766cb12a64fe.png">

- HTML markup, urls, hashtags, unnecessary punctuations and non-ascii digits, whitespaces were removed 
- tokenized with nltk for later machine learning purpose

#### Sample Model Building and Evaluation Process
<img width="644" alt="Screen Shot 2023-01-25 at 12 22 03 AM" src="https://user-images.githubusercontent.com/102776898/214494661-757b1a99-3413-4f6b-a196-b0700fd18ad7.png">
<img width="632" alt="Screen Shot 2023-01-25 at 12 22 47 AM" src="https://user-images.githubusercontent.com/102776898/214494758-fd16a49b-b3fb-4891-9186-d1606b78dcac.png">

- Naive Bayes, Random Forest, Logistic Regression, and Linear Support Vector Machine Model were built using scikit-learn.
- The peformance of each model was evaluated using accuracy score, F1 score and confusion matrix 
- Support Vector Machine model scored the highest accuracy with the pipeline scoring accuracy score of 94.4%
- The SVM model was saved 

#### Running the Web App
Open app.py and run the file using Streamlit by typing in 'streamlit run app.py' in your terminal!

https://user-images.githubusercontent.com/102776898/214497118-8f982649-c5d1-46f3-87b9-f35c4c8d150b.mov

You will be able to view your Streamlit app in your browser.

<h2 id="todo">TODOs </h2>

1. Improve accuracy scores with dealing with imbalanced dataset
