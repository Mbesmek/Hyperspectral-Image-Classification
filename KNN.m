clear; clc; close all; 

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

[w, h, spektral] = size(data);
flatten_data = reshape(data,[w*h,spektral]);%veriyi 200*145*145 lik hale getirdik
gt_vector(:,:) = reshape(ground_truth,1, w*h);


predicted_data_index = ground_truth;

num_of_class = max(ground_truth(:));%kaç sýnýfýmýz var

classSample = zeros(1,num_of_class);%hangi sýnýfta kaç eleman var

for i = 1 : 1 : num_of_class
 classSample(i) = sum(sum((ground_truth == i))); 
end


% bu döngü verilen orana göre train/test verilerini ayýrmak için yazýldý.
for i = 1:1:num_of_class
    
    num_train_sample = fix(classSample(i)*training_data_rate);
    [row,col] = find(ground_truth==i);% konumlarý tutuyor
    
    x = randsrc(1,num_train_sample,1:length(row));
    
    for j=1:1:length(x)
       
        training_data_index(row(x(j)), col(x(j)))= ground_truth(row(x(j)),col(x(j)));%train_data index
        predicted_data_index(row(x(j)), col(x(j))) = 0;
        
    end
end

classNumber_test=zeros(1,num_of_class);
for i=1:1:num_of_class
    classNumber_test(i) = sum(sum(predicted_data_index==i));
end

% toplamda kaç veri svm' girecek.
classNumber_train=zeros(1,num_of_class);
for i=1:1:num_of_class
    classNumber_train(i) = sum(sum(training_data_index==i));
end
number_sample_train = sum(sum(classNumber_train));

[training_data_row, training_data_col] = find(training_data_index ~= 0); % training dalarýn pixel konumlarý
[test_data_row, test_data_col] = find(predicted_data_index ~= 0);

number_sample_test = length(test_data_row);

trainingVector = zeros(number_sample_train, spektral); %svm'e sokmak için veriler vektör haline getiriliyor.
trainingVectorLabel = zeros(number_sample_train,1); %labellarýný verilerle ayný boyutlu bir vektör haline getirdik.
testVector= zeros(number_sample_test, spektral);
testVectorLabel =  zeros(number_sample_test,1);

%%train vektörleri oluþturuldu

for i = 1 : number_sample_train
 trainingVector(i,:) = data(training_data_row(i),training_data_col(i),:);
 trainingVectorLabel(i,1) =ground_truth(training_data_row(i),training_data_col(i));
end

for i = 1 : number_sample_test
 testVector(i,:) = data(test_data_row(i),test_data_col(i),:);
 testVectorLabel(i,1) =ground_truth(test_data_row(i),test_data_col(i));
end

classes = unique(trainingVectorLabel);
num_class = numel(classes);
model = cell(num_class,1);
kernel_type = 'polynomial';

%%%%%%%%%%%%%%% KNN %%%%%%%%%%%%%%%%%%

for i=1:1:size(testVector)
    for j=1:1:size(trainingVector)
        distance(i,j)=sqrt(sum((testVector(i,:)-trainingVector(j,:)).^2));
    end
     [distance(i,:),ind(i,:)]=sort(distance(i,:));
end
k=5;
k_nn=ind(:,1:k);
for i=1:1:size(k_nn,1)
predicted_labels(i,:)=mode((trainingVectorLabel(k_nn(i,:)))');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = zeros(w, h);
for i=1:length(predicted_labels)
    x(test_data_row(i),test_data_col(i)) = predicted_labels(i);
end


%%%%%%%%%% ACCURACY %%%%%%%%%%%%%
%Overal Accuracy
CM = confusionmat(testVectorLabel, predicted_labels);

OverallAcc = trace(CM)/number_sample_test;

KappaAcc = cohensKappa(CM);


for i=1:1:num_of_class
     class_accuracy_mat(i) = CM(i,i)/classNumber_test(:,i);
end

avv_acc = (sum(class_accuracy_mat))/9


figure;
subplot(1,2,1)
imshow(label2rgb(x));title('KNN Output');
subplot(1,2,2)
imshow(label2rgb(ground_truth));title('Ground Truth Image');


