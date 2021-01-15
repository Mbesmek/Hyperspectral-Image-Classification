clear; close all; clc;

training_data_rate = 0.05;

[data,path] = uigetfile('*.mat');
[ground_truth, path_gt] =  uigetfile('*.mat');

data = load(data);
ground_truth = load(ground_truth);

struct_data = fieldnames(data);
data = data.(struct_data{1});

struct_gt = fieldnames(ground_truth);
ground_truth = ground_truth.(struct_gt{1});
% figure;
% subplot(1,2,1)
% imshow(data(:,:,100),[]);title('Indian Pines Hiperspeaktral Image');
% subplot(1,2,2)
% imagesc(uint8(ground_truth));title('Ground Truth Image');
%%
for i=1:220
    data1(:,:,i)=glcm(data(:,:,i));
end
data=data1;
%%
[w, h, spektral] = size(data);
flatten_data = reshape(data,[w*h,spektral]);%veriyi 200*145*145 lik hale getirdik
[w, h, spektral] = size(ground_truth);
gt_vector(:,:) = reshape(ground_truth,w*h,1 );
%figure;
%gscatter(gt_vector,flatten_data);
%training_data_index = zeros(size(ground_truth));%hafýzada yer ayýr train verileri için

predicted_data_index = ground_truth;
num_of_class = max(ground_truth(:));%kaç sýnýfýmýz var


%%% Morp Image-----------------------------------------------------------
dataMorp = MorpOpen(data);

[x_train, y_train, x_test, y_test, test_mat, MorpTrain_x, MorpTest_x] =DataSplit(data, dataMorp, ground_truth, 0.05);


[uniquex, classSample, J]=unique(y_test) ;
cnt = histc(J,unique(J));% her sýnýftan kaç test verisi var

[test_mat_row, test_mat_col] = find(test_mat~=0); % sýnýflandýrma sonuçlarýný bastýrmak için test verilerinin yerini tutar
number_sample_test = length(y_test);

classes = unique(y_train);
num_class = numel(classes);
model = cell(num_class,1);

%% --------------------------------SVM------------------------------------- 
kernel_type = 'Polynomial';
%cvFolds = crossvalind('kfold', x_train, 10);
for k=1:num_of_class
    %for i = 1:10
    %testIdx = (cvFolds == i);                %# get indices of test instances
    %trainIdx = ~testIdx; 
    class_k_label = y_train == classes(k);
    model{k} = fitcsvm(x_train, class_k_label, 'Standardize',...
    true,'KernelScale', 'auto', 'KernelFunction', kernel_type, ...
        'CacheSize', 'maximal', 'BoxConstraint', 10);

end
for k=1:num_of_class
   [predicted_label, temp_score] = predict(model{k}, x_test);
    score(:, k) = temp_score(:, 2);
end

[x_score,y_score]=size(score);
        
[~,max_score] = max(score,[],2); % maximum scorlarý bulduk ve label deðerlerini atadýk.
%x = zeros(w,h);
x = zeros(w,h);


for i=1:length(max_score)
    x(test_mat_row(i),test_mat_col(i)) = max_score(i);
end


%------------- Morphological-------------------------
for k=1:num_of_class
    %for i = 1:10
    %testIdx = (cvFolds == i);                %# get indices of test instances
    %trainIdx = ~testIdx; 
    class_k_label = y_train == classes(k);
    model_M{k} = fitcsvm(MorpTrain_x, class_k_label, 'Standardize',...
    true,'KernelScale', 'auto', 'KernelFunction', kernel_type, ...
        'CacheSize', 'maximal', 'BoxConstraint', 10);
end

for k=1:num_of_class
   [predicted_labelM, temp_scoreM] = predict(model_M{k}, MorpTest_x);
    scoreM(:, k) = temp_scoreM(:, 2);
end

[xM_score,yM_score]=size(scoreM);
        
[~,maxM_score] = max(scoreM,[],2); % maximum scorlarý bulduk ve label deðerlerini atadýk.
%x = zeros(w,h);
xM = zeros(w,h);

for i=1:length(maxM_score)
    xM(test_mat_row(i),test_mat_col(i)) = maxM_score(i);
end

%%%%%%%%%% ACCURACY %%%%%%%%%%%%%
%Overal Accuracy
CM = confusionmat(double(y_test), double(max_score));
Overall_Acc = trace(CM)/number_sample_test;
Kappa_Acc = cohensKappa(CM);
%Average Accuracy
for i=1:1:num_of_class
     class_accuracy_mat(i) = CM(i,i)/cnt(i,:);
end

Avearage_Acc = sum(class_accuracy_mat)/num_of_class;
LastName = {'hiperspektralveri'};

table(LastName, Overall_Acc, Avearage_Acc, Kappa_Acc)
figure;
subplot(1,2,1)
imshow(label2rgb(x));title('SVM Output');
subplot(1,2,2)
imshow(label2rgb(ground_truth));title('Ground Truth Image');

%----------- Morphological Accuracy --------------------------------

CM_M = confusionmat(double(y_test), double(maxM_score));
Overall_AccM = trace(CM_M)/number_sample_test;
Kappa_AccM = cohensKappa(CM_M);
%Average Accuracy

for i=1:1:num_of_class
     class_accuracy_matM(i) = CM_M(i,i)/cnt(i,:);
end

Avearage_AccM = sum(class_accuracy_matM)/num_of_class;
LastName = {'hiperspektralveri'};

table(LastName, Overall_AccM, Avearage_AccM, Kappa_AccM)
figure;
subplot(1,2,1)
imshow(label2rgb(xM));title('Morphological Data SVM Output');
subplot(1,2,2)
imshow(label2rgb(ground_truth));title('Ground Truth Image');

% figure;
% plotconfusion(y_test, maxM_score);


%% ------------------------------- KNN ------------------------------------
for i=1:1:size(x_test)
    for j=1:1:size(x_train)
        distance(i,j)=sqrt(sum((x_test(i,:)-x_train(j,:)).^2));
    end
     [distance(i,:),ind(i,:)]=sort(distance(i,:));
end
k=5;
k_nn=ind(:,1:k);
for i=1:1:size(k_nn,1)
predicted_labels(i,:)=mode((y_train(k_nn(i,:)))');
end

for i=1:length(predicted_labels)
    x_knn(test_mat_row(i),test_mat_col(i)) = predicted_labels(i);
end
figure;
subplot(1,2,1)
imshow(label2rgb(x_knn,  @jet, [.3 .3 .3]));title('KNN Output');
subplot(1,2,2)
imshow(label2rgb(ground_truth,  @jet, [.3 .3 .3]));title('Ground Truth Image');

CM_knn = confusionmat(double(y_test), double(predicted_labels));
Overall_Acc_knn = trace(CM_knn)/number_sample_test;
Kappa_Acc_knn = cohensKappa(CM_knn);
%Average Accuracy

for i=1:1:num_of_class
     class_accuracy_mat_knn(i) = CM_knn(i,i)/cnt(i,:);
end

Avearage_Acc_knn = sum(class_accuracy_mat_knn)/num_of_class;
LastName = {'hiperspektralveri_Knn'};

table(LastName, Overall_Acc_knn, Avearage_Acc_knn, Kappa_Acc_knn)





%% -------------------------- Random Forest -------------------------------

num_tree = ceil(spektral/10); %her 10 özellik için bir karar aðacý oluþtur.
randomForestModel = TreeBagger(num_tree,x_train, y_train,'OOBPrediction','On',...
    'Method','classification', 'NumPredictorsToSample','all');

view(randomForestModel.Trees{1},'Mode','graph');
RD_pred = predict(randomForestModel, x_test);
RD_pred = str2double(RD_pred);
x_rd = zeros(w,h);

for i=1:length(RD_pred)
    x_rd(test_mat_row(i),test_mat_col(i)) = RD_pred(i);
end

%%%%%%%%%% ACCURACY %%%%%%%%%%%%%
%Overal Accuracy

CM_rd = confusionmat(double(y_test), RD_pred);
Overall_Acc_rd = trace(CM_rd)/number_sample_test;
Kappa_Acc_rd = cohensKappa(CM_rd);
%Average Accuracy

for i=1:1:num_of_class
     class_accuracy_mat_rd(i) = CM_rd(i,i)/cnt(i,:);
end

Avearage_Acc_rd = sum(class_accuracy_mat_rd)/num_of_class;
LastName = {'hiperspektralveri(RD)'};

table(LastName, Overall_Acc_rd, Avearage_Acc_rd, Kappa_Acc_rd)
figure;
subplot(1,2,1)
imshow(label2rgb(x_rd,  @jet, [.5 .5 .5]));title('Rondom Forest Output');
subplot(1,2,2)
imshow(label2rgb(ground_truth,  @jet, [.5 .5 .5]));title('Ground Truth Image');

figure;
oobErrorBaggedEnsemble = oobError(randomForestModel);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';

%-----Morphological Data --------------------------------------------------
randomForestModel_M = TreeBagger(num_tree,MorpTrain_x, y_train,'OOBPrediction','On',...
    'Method','classification', 'NumPredictorsToSample','all');

view(randomForestModel_M.Trees{1},'Mode','graph');
RD_pred_M = predict(randomForestModel_M, MorpTest_x);
RD_pred_M = str2double(RD_pred_M);
x_rd_M = zeros(w,h);

for i=1:length(RD_pred_M)
    x_rd_M(test_mat_row(i),test_mat_col(i)) = RD_pred_M(i);
end

%%%%%%%%%% ACCURACY MORP %%%%%%%%%%%%%
%Overal Accuracy

CM_rdM = confusionmat(double(y_test), double(RD_pred_M));
Overall_Acc_rdM = trace(CM_rdM)/number_sample_test;
Kappa_Acc_rdM = cohensKappa(CM_rdM);
%Average Accuracy

for i=1:1:num_of_class
     class_accuracy_mat_rdM(i) = CM_rdM(i,i)/cnt(i,:);
end

Avearage_Acc_rdM = sum(class_accuracy_mat_rdM)/num_of_class;
LastNameM = {'hiperspektralveri(RD)'};

table(LastNameM, Overall_Acc_rdM, Avearage_Acc_rdM, Kappa_Acc_rdM)
figure;
subplot(1,2,1)
imshow(label2rgb(x_rd_M,  @jet, [.5 .5 .5]));title('Rondom Forest Output (Morhological)');
subplot(1,2,2)
imshow(label2rgb(ground_truth,  @jet, [.5 .5 .5]));title('Ground Truth Image');

figure;
oobErrorBaggedEnsembleM = oobError(randomForestModel_M);
plot(oobErrorBaggedEnsembleM)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';