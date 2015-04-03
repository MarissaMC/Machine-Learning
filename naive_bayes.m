function [new_accu, train_accu] = naive_bayes(train_data, train_label, new_data,new_label)
% naive bayes classifier
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  new_data: M*D matrix, each row as a sample and each column as a
%  feature
%  new_label: M*1 vector, each row as a label
%

% Output:
%  new_accu: accuracy of classifying new_data
%  train_accu: accuracy of classifying train_data
%
% CSCI 576 2014 Fall, Homework 1


[train_N,feature_train_N]=size(train_data);
[new_N,feature_new_N]=size(new_data);
class_N=length(unique(train_label));

% to make calculation easier, transform train_label and new_label 
% into binary

train_label_new=zeros(train_N,class_N);
new_label_new=zeros(new_N,class_N);

for i=1:train_N
    if train_label(i)==1
        train_label_new(i,1)=1;
    elseif train_label(i)==2
        train_label_new(i,2)=1;
    elseif train_label(i)==3
        train_label_new(i,3)=1;
    elseif train_label(i)==4
        train_label_new(i,4)=1;
    end
end

for i=1:new_N
    if new_label(i)==1
        new_label_new(i,1)=1;
    elseif new_label(i)==2
        new_label_new(i,2)=1;
    elseif new_label(i)==3
        new_label_new(i,3)=1;
    elseif new_label(i)==4
        new_label_new(i,4)=1;
    end
end

% calculate using Naive Bayes

P_ck=zeros(1,class_N);      % store P(Y=ck)
P_xy=repmat(0,class_N,feature_train_N); % store P(x|y)
P_judge=repmat(0,train_N,class_N);
train_result=repmat(0,train_N,class_N);
new_result=repmat(0,new_N,class_N);

P_ck = sum(train_label_new, 1)/train_N;


for k=1:class_N
    for d=1:feature_train_N
        N_xy=sum(train_data(:,d).*train_label_new(:,k));
        N_y=sum(train_label_new(:,k));
        if N_xy~=0
            P_xy(k,d)=N_xy/N_y;
        elseif N_xy==0;             % in case some features appear in test_data
            P_xy(k,d)=0.001;
        end
    end
end

%count train_accu
for n=1:train_N
    for k=1:class_N
        P_judge(n,k)=P_ck(k);
        for d=1:feature_train_N
            if train_data(n,d)==1
                P_judge(n,k)=P_judge(n,k)*P_xy(k,d);
            end
        end
    end
    for k=1:class_N
        if P_judge(n,k)==max(P_judge(n,:))
            train_result(n,k)=1;
        end
    end
end

%count new_accu
for n=1:new_N
    for k=1:class_N
        P_judge(n,k)=P_ck(k);
        for d=1:feature_new_N
            if new_data(n,d)==1
                P_judge(n,k)=P_judge(n,k)*P_xy(k,d);
            end
        end
    end
    for k=1:class_N
        if P_judge(n,k)==max(P_judge(n,:))
            new_result(n,k)=1;
        end
    end
end

train_accu=sum(sum(train_result.*train_label_new))/train_N
new_accu=sum(sum(new_result.*new_label_new))/new_N

end

