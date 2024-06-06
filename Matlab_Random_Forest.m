clc
clear all
close all
% Importing processed dataset for gender classification
csv = 'C:\Users\user.NBIALPER\Desktop\MACHINE LEARNING\gender_processed';
gender_dataset = readtable(csv);
%% 
% Converting 'gender' column into categorical type in order to use as a binary variable
gender_dataset.gender = categorical(gender_dataset.gender);
% Standardizing 2 numerical variables (forehead_width_cm , forehead_height_cm)
gender_dataset.forehead_width_cm = zscore(gender_dataset.forehead_width_cm);
gender_dataset.forehead_height_cm = zscore(gender_dataset.forehead_height_cm);
%%
% Specifying input (X) and target (Y) variables that will be used in classification
X = gender_dataset(: , 1:end-1);
Y = gender_dataset(:, end);
%%
% Reproducibility provider
rng(1);
% Dividing dataset into test(%20) and train(%80) sets by using HoldOut method
cv_first = cvpartition(size(gender_dataset,1), 'HoldOut', 0.2);
idx = cv_first.test;
x_train = X(~idx, :);
y_train = Y(~idx, :);
x_test = X(idx, :);
y_test = Y(idx, :);
% Checking number of rows 
disp(size(x_train,1));
disp(size(y_train,1));
disp(size(x_test,1));
disp(size(y_test,1));
%% Before performing hyperparameter tuning and cross validation, initializing resultmodel, resultacc, resulthypparameters collectors
% Collecting best results of model, hyperparameters and accuracy to be applied in test data set
 tuned_model = []
 resultacc = 0;
 resulthypparameters = struct();
% Selecting 'minimum leaf size' and 'maximum number of splits' as hyperparameters to be tuned
 hypparameters = struct(...
     'NumTrees', [50, 100], ... 
     'MinLeafSize', [5, 10], ...
     'MaxNumSplits', [10, 15] ...
     );
%%
% Second split on training set to apply cross validation and hyperparametertuning at the same time
fold_number = 5;
cv_second = cvpartition(size(x_train,1), 'KFold', fold_number);
for f = 1:fold_number
    second_train_idx = training(cv_second, f);
    val_idx = test(cv_second, f);
    x_second_train = x_train(second_train_idx, :);
    y_second_train = y_train(second_train_idx, :);
    x_val = x_train(val_idx, :);
    true_val = y_train(val_idx, :);
    %For Random Forest method TreeBagger model is used 
    for min_leaf_size = hypparameters.MinLeafSize
        for max_split_number = hypparameters.MaxNumSplits
            for tree_number = hypparameters.NumTrees
            %Applying the current hyperparameters on TreeBagger model and predicting on validation data set 
            model = TreeBagger(tree_number, x_second_train, y_second_train, ...
                'Method','classification', ...
                'NumPredictorsToSample','all', ...
                'MaxNumSplits', max_split_number, ...
                'MinLeafSize', min_leaf_size ...
                );
            model_predicted = predict(model, x_val);
            %Converting 'model_predicted' into categorical type in order to calculate accuracy
            model_predicted = categorical(model_predicted);
            y_val = true_val.gender
            model_accuracy = sum (y_val == model_predicted) / numel(y_val);
            %Calculating Confusion matrix 
            model_conf = confusionmat(y_val, model_predicted)
            %If the accuray boosted update hyperparameter
            if model_accuracy > resultacc
                resultacc = model_accuracy;
                tuned_model = model;
                resulthypparameters = struct('NumTrees', tree_number, 'MaxNumSplits', ...
                    max_split_number, 'MinLeafSize', min_leaf_size);
            end
        end
    end
    end
end
best_parameters = (resulthypparameters);
disp(best_parameters);
best_accuracy = (resultacc);
disp(best_accuracy);
%%
%Applying the best hyperparameters on the entire train set
tuned_model = TreeBagger(resulthypparameters.NumTrees, x_train, y_train, ...
                'Method','classification', ...
                'NumPredictorsToSample','all', ...
                'MaxNumSplits', resulthypparameters.MaxNumSplits, ...
                'OOBPrediction', 'on', ...
                'OOBPredictorImportance', 'on');
%Predicting tuned_model on the test set
tuned_predicted = predict(tuned_model, x_test);
% Converting 'result_predicted' into categorical type in order to calculate accuracy
tuned_predicted = categorical(tuned_predicted);
y_test = y_test.gender;
test_accuracy = sum (y_test == tuned_predicted) / numel(y_test);
%% save test data
save('decision_tree.mat', 'y_test');
%%
dt_conf = confusionmat(y_test, tuned_predicted);
figure;
confusionchart(y_test, tuned_predicted, 'ColumnSummary', 'column-normalized', ...
    'RowSummary', 'row-normalized', 'DiagonalColor', 'magenta');
title ('Gender Classification (RF)');
TP = dt_conf(1, 1);
FP = dt_conf(1, 2);
FN = dt_conf(2, 1);
Precision = TP / (TP + FP);
Recall = TP / (TP + FN);
F1_score = 2 * (Precision * Recall) / (Precision + Recall);
%%
importance_bar = tuned_model.OOBPermutedVarDeltaError;
[importance_Sorted, idx] = sort(importance_bar, 'descend');
features = gender_dataset.Properties.VariableNames;
features = categorical(features);
featuresSorted = features(idx);
featuresSorted = cellstr(featuresSorted)
importance_percentage_rf_sorted = (importance_Sorted / sum(importance_Sorted)) * 100;
Color_selection = jet(length(featuresSorted));
   
figure;
b = bar(importance_percentage_rf_sorted, 'FaceColor', 'flat', 'CData', Color_selection);
colormap(Color_selection );
xticks(1:length(featuresSorted));
xticklabels(featuresSorted);
xlabel('Features');
ylabel('Importance Percentage');
title('Feature Importance (RF)');
