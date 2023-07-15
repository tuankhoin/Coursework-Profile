rng(1025294);
data = load('lidarData.mat');
data = data.lidarData;
lbl = load('lidarLabel.mat');
lbl = lbl.lidarLabel;
%% 3.1-3.2

l = length(data);
feats = zeros(l,6);
pf = zeros(l,3);
for i=1:l
    c = data{1,i};
    % 3.1 Intensity-based features
    A = c(:,4);
    feats(i,1) = mean(A);
    feats(i,2) = std(A);
    feats(i,3) = median(A);
    % 3.2 Max spread features
    feats(i,4) = spread(c(:,1));
    feats(i,5) = spread(c(:,2));
    feats(i,6) = spread(c(:,3));
    % 3.5 PCA feature extraction
    [~,~,lat] = pca(c(:,1:3));
    pf(i,:) = lat';
end
f31 = feats(:,1:3);
f32 = feats(:,4:6);
%% 3.3

rng(1025294);
% Get indexing list for the partition
cv = cvpartition(lbl,'HoldOut',0.3,'Stratify',true);
idx = cv.test();
test_idx = find(idx);
train_idx = find(~idx);

% Assign datasets according to the index list
train_x = feats(train_idx,:);
test_x = feats(test_idx,:);
train_y = {lbl{train_idx}};
test_y = {lbl{test_idx}};

% Ensure that all labels appeared in each dataset
train_labels = unique(train_y)
test_labels = unique(test_y)
%% 3.4

fprintf('Training with 3 spread features and 3 intensity features');
acc = svm_score(train_x,train_y,test_x,test_y,train_labels)
%% 3.5

% 0R as baseline
lbl_count = countcats(categorical(lbl));
fprintf('0R Baseline - Proportion of the most common label in the whole dataset')
baseline = max(lbl_count) / sum(lbl_count)
%%
% 3 intensity features
train_i = train_x(:,1:3);
test_i = test_x(:,1:3);
fprintf('Training with 3 intensity features');
acc = svm_score(train_i,train_y,test_i,test_y,train_labels)
%%
% Adding PCA extracted features
for i = 1:3
    fprintf('Adding %d extra PCA feature(s)',i);
    svm_score([train_i pf(train_idx,1:i)],train_y, [test_i pf(test_idx,1:i)],test_y,train_labels)
end
%%
function s = spread(L)
    s = max(L) - min(L);
end

function acc = svm_score(train_x,train_y, test_x,test_y,train_labels)
    mdl = fitcecoc(train_x,train_y,'Learners',templateSVM('Standardize',true,'KernelFunction','linear'), ...
                                   'ClassNames',train_labels, ...
                                   'Verbose',0);
    pred_y = predict(mdl,test_x);
    correct = strcmpi(pred_y,test_y');
    acc = sum(correct)/length(correct);
end