# NLP-Model-Comparison
This method was developed to address challenges encountered when applying NLP models to contextual-level sentiment analysis of text data. We were analysing undergraduate students' open-ended responses to survey questions. First, the responses often included incomplete sentences, informal language, and grammatical errors. 

Second, since NLP models are pre-trained, they can have training data bias. For example, in our data, the term “dictator,” used by a student to describe their instructor, was considered a neutral sentence by most of the models we used. However, this interpretation was incorrect for our responses to the open-ended question. You can read more about the nature of the dataset in <a href= "https://www.mdpi.com/2227-7102/16/3/457">the paper published in Education Sciences</a>. 

Here are the steps to define the best NLP model for your dataset. All code for these steps is in the NLP_model_comparison.py file, and each step is marked with a comment so you know where the code starts.

**Step 1**

To determine the best NLP model for contextual-level sentiment analysis, given the overall sample size, you need to calculate how many samples are required for manual testing. We used <a href="https://ask.ifas.ufl.edu/publication/PD006">Equation 6 described by Israel (1992)</a>. Based on the calculation, we randomly selected 334 statements from the data. You can use the code below in Python to select samples. It's also in the NLP_model_comparison.py file. 

```sh
# Select a random sample of 310 rows
sample_df = xt.sample(n=310, random_state=42)

sample_df['your_data'] = np.nan
sample_df.drop('Study ID', axis=1, inplace=True) #Study ID is a unique number for each participant  

# Save the sample to a new CSV file if needed
# sample_df.to_excel(r"C:name_of_your_repository\random_sample310.xlsx", index=False)

sample_df = pd.read_excel(r"C:name_of_your_repository\random_sample310.xlsx")
```

**Step 2**
Randomly selected sample statements were manually coded for positive and negative sentiment. A neutral category was not included because not all NLP models support it; therefore, the analysis was limited to two sentiment classes. Additionally, a single coder labelled statements, since the purpose of manual coding was model validation rather than primary analysis. However, if you have a partner, you can both label statements manually and then compare them to each other to achieve an 80% reliability score. 

**Step 3**
Now we need to prepare the 334 manually labelled statements for the analysis. The first step is to clean that data, remove stopwords, lemmatise sentences, and remove punctuations and numbers. Codes are in the NLP_model_comparison.py file.

**Step 4**
Now use the cleaned sample statements to test against 43 NLP models. You can expand the list to suit your preferences. As success metrics, we recorded the accuracy and F1 scores for each model, aligned with our manual sentiment analysis. 

In our case, the Robustly optimized BERT approach (RoBERTa) Large English library had the highest F1-Score (92%) and accuracy (92%) based on our data. Therefore, we selected RoBERTa for our sentiment analysis as it had the highest fit scores.

**Visualisation of success metrics**

In our study, we used a Radar Chart to visualise the results of success metrics in an appealing way. All the codes are in the NLP_comparison_radar_chart.py file. You can adjust the positions of some labels to avoid overlapping in names. All these instructions are described in the code. The result will look like this:

<div align="center">
  <img width="600" alt="Visualisation of the performance of 43 models, highlighting F1 and accuracy scores" src="https://github.com/elnaramedia/NLP-Model-Comparison/blob/2cb092392c9ba432a46155461b18a5614501ea8b/Model%20Performance%20radar%20chart.png" />
</div>


