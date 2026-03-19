

import pandas as pd
import numpy as np
import glob
import os
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1500)

import matplotlib
matplotlib.use("Agg")



df = pd.read_excel(r"your_data.xlsx")


#### #### #### #### #### #### ####
#### Sample size selection =334 ##
#### #### #### #### #### #### ####

#STEP 1
# Select a random sample of 334 rows
sample_df = xt.sample(n=334, random_state=42)

sample_df['your_data'] = np.nan
sample_df.drop('Study ID', axis=1, inplace=True) #Study ID is a unique number for each participant  

# Save the sample to a new CSV file if needed
# sample_df.to_excel(r"C:name_of_your_repository\random_sample334.xlsx", index=False)

sample_df = pd.read_excel(r"C:name_of_your_repository\random_sample334.xlsx")

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Now define the best NLP model based on manual coded 334 dataset. DATA PREPROCESSING   ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

# STEP 3

import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Initialize the stopwords and lemmatizer
sw = set(stopwords.words('english'))
sw.add('us')
lemmatizer = WordNetLemmatizer()

# get_wordnet_pos helps the performance of lemmatizetion.
# Function to convert NLTK POS tags to WordNet POS tags
# def get_wordnet_pos(word):
#     tag = pos_tag([word])[0][1][0].upper()
#     tag_dict = {"J": wordnet.ADJ,
#                 "N": wordnet.NOUN,
#                 "V": wordnet.VERB,
#                 "R": wordnet.ADV}
#     return tag_dict.get(tag, wordnet.NOUN)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Function to remove stopwords from a sentence
def remove_stopwords(sentence):
    return " ".join([word for word in sentence.split() if word.lower() not in sw])

# Function to lemmatize a sentence with POS tagging
# def lemmatize_sentence(sentence):
#     word_tokens = word_tokenize(sentence)
#     lemmatized_sentence = []
#     pos_tagged_tokens = pos_tag(word_tokens)
#     for word, pos in pos_tagged_tokens:
#         wordnet_pos = get_wordnet_pos(word)
#         lemmatized_sentence.append(lemmatizer.lemmatize(word, wordnet_pos))
#     return " ".join(lemmatized_sentence)


def lemmatize_sentence(sentence):
    word_tokens = word_tokenize(sentence)
    pos_tagged_tokens = pos_tag(word_tokens)
    lemmatized_sentence = []
    for word, tag in pos_tagged_tokens:
        # Correct POS tag for "allows" and "think"
        if word.lower() == 'allows' or word.lower() == 'thinks':
            tag = 'VB'
        wordnet_pos = get_wordnet_pos(tag)
        lemmatized_word = lemmatizer.lemmatize(word, wordnet_pos)
        lemmatized_sentence.append(lemmatized_word)
    return " ".join(lemmatized_sentence)


### Normalizing ###
### Q11l is the name of the column where 310 samples are stored
sample_df['Q11l'] = sample_df['Question'].astype(str).str.lower()

### Removing Punctuations ###
sample_df['Q11l'] = sample_df['Q11l'].str.replace(r'[^\w\s]', '', regex=True)

### Removing Numbers ###
sample_df['Q11l'] = sample_df['Q11l'].str.replace(r'\d', '', regex=True)

### Removing Stopwords ###
sample_df['Q11l'] = sample_df['Q11l'].apply(remove_stopwords)

### Lemmatization ###
sample_df['Q11l'] = sample_df['Q11l'].apply(lemmatize_sentence)

# Display the DataFrame
print(sample_df)


#### #### #### #### #### #### #### #### #### #
### Now run the comparison model analysis ####
#### #### #### #### #### #### #### #### #### #

# STEP 4

from transformers import pipeline, set_seed
from textblob import TextBlob
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Set the seed for reproducibility
seed = 1
set_seed(seed)

# Create a list of sentences
sentences = sample_df['Q11l'].to_list()


# Load different sentiment analysis models
# Extend your list of models
models = [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "textattack/bert-base-uncased-SST-2",
    "nlptown/bert-base-multilingual-uncased-sentiment",
    "cardiffnlp/twitter-roberta-base-sentiment",
    "bhadresh-savani/bert-base-uncased-emotion",
    "siebert/sentiment-roberta-large-english",
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    "finiteautomata/bertweet-base-sentiment-analysis",
    "unitary/toxic-bert",
    "distilbert-base-uncased",
    "roberta-base",
    "textattack/roberta-base-SST-2",
    "xlnet-base-cased",
    "albert-base-v2",
    "facebook/bart-large",
    "bert-base-uncased",
    "google/electra-base-discriminator",
    "vinai/bertweet-base",
    "textattack/distilbert-base-uncased-SST-2",
    "j-hartmann/emotion-english-roberta-large",
    "j-hartmann/emotion-english-distilroberta-base",
    "textattack/bert-base-uncased-imdb",
    "microsoft/deberta-base",
    "microsoft/deberta-large",
    "microsoft/deberta-xlarge",
    "roberta-large",
    "distilroberta-base",
    "textattack/distilroberta-base-SST-2",
    "distilbert-base-uncased-distilled-SST-2",
    "distilbert-base-uncased-emotion",
    "j-hartmann/bert-emotion",
    "j-hartmann/roberta-emotion",
    "finiteautomata/bertweet-large-sentiment",
    "finiteautomata/roberta-large-sentiment",
    "siebert/sentiment-roberta-large",
    "siebert/sentiment-roberta-base",
    "siebert/roberta-large-SST-2",
    "siebert/roberta-base-SST-2",
    "textattack/roberta-large-imdb",
    "bert-large-uncased",
    "bert-large-cased",
    "xlnet-large-cased",
    "albert-large-v2",
    "albert-xlarge-v2",
    "distilbert-base-cased",
    "google/electra-large-discriminator",
    "t5-large",
    "openai-gpt",
    "gpt2-large",
    "roberta-large-mnli",
    "facebook/bart-large-cnn",
    "facebook/bart-large-mnli",
    "nghuyong/ernie-2.0-en",
    "allenai/longformer-large-4096",
    "microsoft/deberta-v2-xlarge",
    "microsoft/deberta-v2-xxlarge",
    "Helsinki-NLP/opus-mt-en-de",
    "xlm-roberta-large",
    "facebook/mbart-large-cc25",
    "textblob"
]


# Initialize results dictionary
results = []

# Analyze sentences with each model
for model_name in models:
    try:
        model = pipeline("sentiment-analysis", model=model_name)
        for sentence in sentences:
            try:
                result = model(sentence)[0]
                results.append({'Model': model_name, 'Sentence': sentence, 'Label': result['label'], 'Score': result['score']})
            except Exception as e:
                print(f"Error analyzing sentence with model {model_name}: {e}")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")

# Create a DataFrame to store the results
df_results = pd.DataFrame(results)

df_results['Label'].drop_duplicates()
df_results.groupby('Model')['Label'].unique()
df_results[df_results['Model']=='xlnet-large-cased']

# Reshape the DataFrame so sentences are rows and models are columns
df_melted = df_results.pivot_table(index='Sentence', columns='Model', values=['Label', 'Score'], aggfunc='first')
df_melted.columns = [f'{col[0]}_{col[1]}' for col in df_melted.columns]  # Flatten the multi-index columns



#### #### #### #### #### #### #### #### #### #### #### #### #### #
### Standardize sentiment labels into 3-class sentiment scale ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #

def standardize_sentiment_label(label):
    positive_labels = [
        "POSITIVE", "LABEL_1", "4 stars", "5 stars", "surprise", "joy", "love", "positive", "POS", "ENTAILMENT", "entailment"]
    neutral_labels = [
        "3 stars", "neutral", "NEU", "NEUTRAL", "LABEL_2"]
    negative_labels = [
        "NEGATIVE", "LABEL_0", "2 stars", "1 star",  "anger", "fear", "negative", "NEG",
        "toxic", "sadness", "disgust", "CONTRADICTION", "contradiction"]

    if label in positive_labels:
        return 'positive'
    elif label in neutral_labels:
        return 'neutral'
    elif label in negative_labels:
        return 'negative'


df_standardized = df_melted.applymap(standardize_sentiment_label)
df_standardized.info()
df_standardized = df_standardized.iloc[:, :44]


#### #### #### #### #### #### #### #
### Let's get the true Y values ####
#### #### #### #### #### #### #### #

elnara = pd.read_excel(r"C:your_repository_name\random_sample310.xlsx")

elnara['Elnara'].unique() #'Elnara' is the column name where I coded statements as positive ('P') and negative ('N')  


sample_df1 = pd.merge(sample_df.rename(columns={'Q11':'Question'}).drop('Elnara', axis=1).drop_duplicates(subset='Q11l'), elnara.drop_duplicates(subset='Question'), on='Question', how='left')

def clarify(value):
    if value == 'p':
        return 'positive'
    elif value == 'n':
        return 'negative'
    elif pd.isnull(value):
        return 'neutral'
    else:
        return 'unknown'  # Handle any unexpected values

# Apply the function to the 'Elnara' column and create a new column for the sentiment classification
sample_df1['Elnara1'] = sample_df1['Elnara'].apply(clarify)
sample_df1['Elnara1'].unique()

def clarify2(value):
    if value == 'n':
        return 'negative'
    else:
        return 'positive'
 # Handle any unexpected values

# Apply the function to the 'Elnara' column and create a new column for the sentiment classification
sample_df1['Elnara2'] = sample_df1['Elnara'].apply(clarify2)



sample_df2 = pd.merge(sample_df1, df_standardized, how='left', left_on='Q11l', right_on='Sentence')
sample_df2.columns



#### #### #### #### #### #### 
####  SUCCESS METRICS    #### 
#### #### #### #### #### ####

# Initialize a dictionary to store metrics for each model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
# sample_df2['Elnara1'] = sample_df2['Elnara1'].map(label_mapping)


model_columns = sample_df2.iloc[:, 5:].columns


label_mapping = {'negative': 0, 'neutral': 1, 'positive': 1}
sample_df2['Elnara2'] = sample_df2['Elnara2'].map(label_mapping)

# sample_df2[sample_df2[model_columns].isnull()].fillna(0, inplace=True)

for column in model_columns:
    sample_df2[column] = sample_df2[column].map(label_mapping)

metrics = {}

# Drop rows with NaN values in 'Elnara1' and in each model column
for column in model_columns:
    # Combine the columns to check for NaNs
    combined = pd.concat([sample_df2['Elnara2'], sample_df2[column]], axis=1)
    combined.dropna(inplace=True)

    y_true = combined['Elnara2']
    y_pred = combined[column]

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)  ## Manual coding result: y_true, NLP model outputs y_pred

    # Store the metrics
    metrics[column] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': conf_matrix
    }

# Convert metrics dictionary to DataFrame for better visualization
metrics_df = pd.DataFrame(metrics).transpose()

# If you want to sort by F1-Score and display the confusion matrices separately
metrics_df_sorted = metrics_df.sort_values(by='F1-Score')
print(metrics_df_sorted)

# Display confusion matrices
for model in metrics_df_sorted.index:
    print(f"\nConfusion Matrix for {model}:\n{metrics_df_sorted.loc[model, 'Confusion Matrix']}")


#### #### #### #### #### ####  
####    RADAR CHART      ####
#### #### #### #### #### ####



