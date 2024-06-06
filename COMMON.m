clc
clear all
close all
%random forest and decision tree metrics
rf_precision = 0.9733;
rf_recall = 0.9753;
rf_f1_score = 0.9743;
dt_precision = 0.9548;
dt_recall = 0.9708;
dt_f1_score = 0.9627;
%%
%Visualizing metrics in a table
precision_values = [0.9733, 0.9548];
recall_values = [0.9753, 0.9708];
f1_score_values = [0.9743, 0.9627];
models = {'Random Forest', 'Decision Tree'};
data = table(precision_values', recall_values', f1_score_values', ...
    'VariableNames', {'Precision', 'Recall', 'F1_Score'}, 'RowNames', models);
uitable('Data', table2cell(data), 'ColumnName', data.Properties.VariableNames, ...
    'RowName', data.Properties.RowNames, 'Units', 'Normalized', 'Position', [0, 0, 1, 1]);