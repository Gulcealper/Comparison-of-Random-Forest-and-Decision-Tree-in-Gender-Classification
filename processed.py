# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 01:03:54 2023

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('gender_classification_v7.csv')

#check properties of table and variables

df.info()

#check missing values

df.isna().sum()  #no missing values

#summary statistics and plots for numeric variables


var1=df['forehead_width_cm']
var2=df['forehead_height_cm']
summary = pd.DataFrame({
    'Forehead Width': [var1.mean(), var1.median(), var1.std(), var1.max()],
    'Forehead Height': [var2.mean(), var2.median(), var2.std(), var2.max()]
}, index=['Mean', 'Median', 'Standard Deviation', 'Maximum'])
summary.plot(kind='bar', rot=0, colormap='Spectral', figsize=(8, 8))
plt.ylabel('Value', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.show()


#Distributions with kernel density

plt.rcParams.update({'font.size': 20})  
custom_palette = {'Male': 'magenta', 'Female': 'slateblue'}
ax = sns.displot(df, x='forehead_width_cm', hue='gender', kind='kde', fill=True, palette=custom_palette)
ax.set_axis_labels(x_var='', y_var='')
ax.fig.suptitle('Distribution of Forehead Width by Gender', y=0.98)
plt.show()


plt.rcParams.update({'font.size': 20})  
custom_palette = {'Male': 'magenta', 'Female': 'slateblue'}
ax = sns.displot(df, x='forehead_height_cm', hue='gender', kind='kde', fill=True, palette=custom_palette)
ax.set_axis_labels(x_var='', y_var='')
ax.fig.suptitle('Distribution of Forehead Height by Gender', y=0.98)
plt.show()



#plots for binary variables

binary_variables = df[['long_hair', 'nose_wide', 'nose_long', 
                      'lips_thin', 'distance_nose_to_lip_long', 'gender']]

integer_variables = binary_variables.columns[binary_variables.dtypes == 'int64']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))
axes = axes.flatten()

for k, variable in enumerate(integer_variables):
    counts = binary_variables[variable].value_counts()
    axes[k].pie(counts, labels=None, autopct=lambda p: '{:.1f}%'.format(p), textprops={'fontsize': 25}, colors=['lightgreen', 'mediumslateblue'])
    axes[k].set_title(variable, fontsize=25)
    axes[k].legend(labels=counts.index, loc='upper right', fontsize=15)

gender_counts = binary_variables['gender'].value_counts()
axes[-1].pie(gender_counts, labels=None, autopct=lambda p: '{:.1f}%'.format(p), textprops={'fontsize': 25}, colors=['lightgreen', 'mediumslateblue'])
axes[-1].set_title('Gender', fontsize=25)
axes[-1].legend(labels=gender_counts.index, loc='upper right', fontsize=15)

plt.tight_layout(pad=0)

plt.show()



#check correlations

df['gender_num'] = df['gender'].apply(lambda x: 0 if x == 'Male' else 1)
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", 
                      annot_kws={"size": 20}, xticklabels=correlation_matrix.columns, 
                      yticklabels=correlation_matrix.columns)
plt.xticks(rotation=0)
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=20)  
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=20)
heatmap.collections[0].colorbar.ax.tick_params(labelsize=20)
plt.title('Correlation Matrix Heatmap', fontsize=25)
plt.show()
















