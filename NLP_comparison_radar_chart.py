

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

#### #### #### #### #### ####  
####    RADAR CHART      ####
#### #### #### #### #### ####

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
# Use the Agg backend
matplotlib.use('Qt5Agg')
import string
from math import pi
# Reset index and rename for merging
# metrics_df1 = metrics_df.reset_index().rename(columns={'index': 'Model'})
metrics_df1 = metrics_df.reset_index().rename(columns={'Unnamed: 0': 'Model'})

# Merge metrics with short names
metrics_df1 = pd.merge(metrics_df1, smnames.rename(columns={'Original Name': 'Model'}), how='left', on='Model')

# Calculate the average of Accuracy, Precision, and Recall
# metrics_df1['Avg_APR'] = metrics_df1[['Accuracy', 'Precision', 'Recall']].mean(axis=1)

# Keep relevant columns and sort by F1-Score
metrics_df1 = metrics_df1[['Short Name', 'F1-Score', 'Accuracy']].drop_duplicates(subset=['Short Name']).sort_values(by='F1-Score', ascending=False)

# Number of variables
categories = metrics_df1['Short Name']
N = len(categories)

# Angles for each axis
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialize the spider plot
plt.figure(figsize=(9, 9))
ax = plt.subplot(111, polar=True)

# Draw one axe per variable + add labels
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], [])

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="black", size=12)
plt.ylim(0, 1)

# Plot F1-Score
values_f1 = metrics_df1['F1-Score'].tolist()
values_f1 += values_f1[:1]
ax.plot(angles, values_f1, linewidth=1, linestyle='solid', label='F1-Score')
ax.fill(angles, values_f1, 'b', alpha=0.1)

# Plot Avg_APR
values_apr = metrics_df1['Accuracy'].tolist()
values_apr += values_apr[:1]
ax.plot(angles, values_apr, linewidth=1, linestyle='solid', label='Accuracy', color='orange')
ax.fill(angles, values_apr, 'orange', alpha=0.1)



for i, angle in enumerate(angles[:-1]):
    angle_rad = angles[i]
    ha = 'center'
    distance = 1.1  # Distance from the center

    if 0 <= angle_rad < pi / 2:
        ha = 'left'
    elif pi / 2 <= angle_rad < pi:
        ha = 'left'
    elif pi <= angle_rad < 3 * pi / 2:
        ha = 'right'
    else:
        ha = 'right'

    ax.text(angle_rad, distance, categories.iloc[i], size=10, horizontalalignment=ha, verticalalignment='center')


# Show the legend
plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1.1), title='Metrics')

# Show the plot
plt.show()





# Avoid overlapping
metrics_df1 = metrics_df.reset_index().rename(columns={'Unnamed: 0': 'Model'})

# Merge metrics with short names
metrics_df1 = pd.merge(metrics_df1, smnames.rename(columns={'Original Name': 'Model'}), how='left', on='Model')
metrics_df1.loc[28, 'Short Name'] = 'BERT Multilingual'
# Keep relevant columns and sort by F1-Score
metrics_df1 = metrics_df1[['Short Name', 'F1-Score', 'Accuracy']].drop_duplicates(subset=['Short Name']).sort_values(by='F1-Score', ascending=False)

# Number of variables
categories = metrics_df1['Short Name']
N = len(categories)

# Angles for each axis
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialize the spider plot
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# Draw one axe per variable + add labels
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], [])

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="black", size=12)
plt.ylim(0, 1)

# Plot F1-Score
values_f1 = metrics_df1['F1-Score'].tolist()
values_f1 += values_f1[:1]
ax.plot(angles, values_f1, linewidth=1, linestyle='solid', label='F1-Score')
ax.fill(angles, values_f1, 'b', alpha=0.1)

# Plot Accuracy
values_acc = metrics_df1['Accuracy'].tolist()
values_acc += values_acc[:1]
ax.plot(angles, values_acc, linewidth=1, linestyle='solid', label='Accuracy', color='orange')
ax.fill(angles, values_acc, 'orange', alpha=0.1)


# # Custom positions for specific labels

# Custom positions for specific labels
custom_labels = {
    'DeBERTa Large': (1.16, 'center', 'top'),
    'Roberta Large English': (1.2, 'center', 'top'),
    'Roberta Twitter Sentiment': (1.1, 'left', 'top'),
    'BERT Multilingual': (1.15, 'left', 'top'),
    'BERT IMDB': (1.2, 'center', 'top'),
    'ALBERT XLarge v2': (1.16, 'center', 'top'),
    'ALBERT Base v2': (1.1, 'left', 'top'),
    'Emotion DistilRoberta': (1.15, 'left', 'top'),
    'Bertweet Sentiment': (1.1, 'left', 'top'),
    'DistilRoberta Finance': (1.05, 'left', 'top'),
}

# Add labels
for i, angle in enumerate(angles[:-1]):
    angle_rad = angles[i]
    ha = 'center'
    va = 'top'
    distance = 1.17  # Default distance from the center

    label = categories.iloc[i]
    if label in custom_labels:
        distance, ha, va = custom_labels[label]

    ax.text(angle_rad, distance, label, size=10, horizontalalignment=ha, verticalalignment=va)


# # Add labels
# for i, angle in enumerate(angles[:-1]):
#     angle_rad = angles[i]
#     ha = 'center'
#     distance = 1.15  # Distance from the center
#     if angle_rad == 0 or angle_rad == pi:
#         ha = 'center'
#     elif 0 < angle_rad < pi:
#         ha = 'center'
#     else:
#         ha = 'center'
#     ax.text(angle_rad, distance, categories.iloc[i], size=10, horizontalalignment=ha, verticalalignment='top')


# Show the legend
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title='Metrics')

# Show the plot
plt.show()






