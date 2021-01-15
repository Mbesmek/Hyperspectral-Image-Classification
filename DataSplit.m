function [trainx,trainy,testx,testy,test_mat, trainxM, testxM ]=DataSplit(data,MorpData,ground_truth,rate)

[w1, h1, spektral] = size(data);
flatten_x = reshape(data,[w1*h1,spektral]);
flatten_y(:,:) = reshape(ground_truth,w1*h1,1 );

flatten_xM = reshape(MorpData,[w1*h1,spektral]);

indx = find(flatten_y ~= 0);
datay = flatten_y(indx,:);
datax = flatten_x(indx,:);

classNum = length(unique(flatten_y));

for i = 1 : 1 : classNum
 classSample(i) = sum(sum((flatten_y == i)));
end

for i = 1:1:classNum
    num_train_sample = fix(classSample(i)*rate);
    [row,col] = find(flatten_y==i); % konumları tutuyor
    if i==1
        output_train(:,i) = row(randperm(numel(row),num_train_sample));
    else
        output_train(end+1:end+num_train_sample,:) = row(randperm(numel(row),num_train_sample));
    end
end
x = 1:1:length(flatten_y);
r = randperm(size(output_train,1));
output_train = output_train(r,:);

x =reshape(x,length(flatten_y), 1);
[~,idx]=ismember(x, output_train);

[w,h] =size(output_train);
train_data_index = reshape(output_train,w*h,1);

for i=1:length(train_data_index)
    trainx(i,:) = flatten_x(train_data_index(i),:);
    trainy(i,:) = flatten_y(train_data_index(i),:);
    trainxM(i,:) = flatten_xM(train_data_index(i),:);

end


testindx = find(idx==0);
testx = flatten_x(testindx, :);
testy = flatten_y(testindx);

testxM = flatten_xM(testindx, :);

test_mat =zeros(length(flatten_y),1);
test_mat(testindx) = flatten_y(testindx,:);
test_mat = reshape(test_mat, w1,h1);

test_idx = find(testy~=0);
testx = testx(test_idx,:);
testy = testy(test_idx,:);
testxM = testxM(test_idx,:);
