sigmoid = @(x) (1./(1+exp(-x)));
hotone = @(X)bsxfun(@eq, X(:), 1:10);%to onehot
funNormalize = @(x) ( x-min(min(x(:))))/( max(max(x))-min(min(x)) + eps);
%------------------------data preprocess------------------------
% load('data_feature_test.mat');
% load('data_feature_train.mat'); 
% data_train_Y_orig = h5read('train_dataset.h5','/train_labels');
% data_test_Y_orig = h5read('test_dataset.h5','/test_labels');
% data_train_Y_preprocessed=hotone(data_train_Y_orig+1);
% data_test_Y_preprocessed=hotone(data_test_Y_orig+1);
% data_train_Y_preprocessed(10000,16)=0;
% data_test_Y_preprocessed(10000,16)=0;

%% ------------------------train------------------------
%here forw_propV3 and back_propV2 are choosen
m1=1024;m2=16;
c=[2,2,2,2,1,1,1,1,1,1];

batch_size=10;
epoch=1;
size_train=60000;
% size_test=1000;
catch_cost=[];
catch_cost_test=[];
% a=10^(-1/t_end);
a=0.996;
%----------------these parameters should be closed if starting frme last point--------
% U=rand(2,2,(m1/2),int8(log2(m1)));
% load('U.mat');
t_end=1000;
% t_end=epoch*size_train/batch_size;
% learning_rate=0.01;

[catch_para_layers,grap]=get_classes(c,m1);
ind=grap(:,end).*(1:m1)';
for i=1:t_end
%     index=(1:batch_size)+batch_size*(i-1)+1000;
    index=randperm(batch_size)+batch_size*(i-1)+1000;
    train_temX=train(index,:);
    train_temY=data_train_Y_preprocessed(index,:);
    sum_delt_U=0;
    for j=1:batch_size
    X=funNormalize(train_temX(j,:));
    X(isnan(X))=0;
    d=train_temY(j,:);
    [catch_X]=forward_propV3(catch_para_layers,grap,X,U);
    catch_X(:,end)=catch_X(:,end).*grap(:,end);
    delt_U=back_propV2(catch_X,catch_para_layers,grap,U,d);
    sum_delt_U=sum_delt_U+delt_U;
%     sum(delt_U(:))
    %-----end------------
    end
    Y=catch_X(:,end);
    d1=zeros(m1,1);
    d1(ind(ind>0))=d;
    cost=sum((d1-Y).^2);
    catch_cost=[catch_cost,cost];
    learning_rate=learning_rate*a;
    U=U-learning_rate*sum_delt_U/batch_size;
    %----early stop------
    if cost<0.1
        break
    end
    %cost of test--------
    if rem(i,10000000)==0
        sum_cost_test=0;
        for k=1:size_test
            X=testset(k,1:m1);
            d_input=testset(k,m1+1:end)';
            d=zeros(m1,1);ind=grap(:,end).*(1:m1)';
            d(ind(ind>0))=d_input;
            [catch_X]=forward_propV3(catch_para_layers,grap,X,U);
            Y=catch_X(:,end);
            cost_test=sum((d-Y).^2);
            sum_cost_test=sum_cost_test+cost_test;
        end
    cost_test=sum_cost_test/size_test;
    catch_cost_test=[catch_cost_test,cost_test];
    end
    if rem(i,10)==0
        fprintf('the cost of %d iter is %f\n',i,cost)
    end
end
% figure(),plot(1:1:t_end,catch_cost(1:1:end),'r',1:100:t_end,catch_cost_test,'g');
save('U.mat','U');
plot(catch_cost(1:1:end),'r')
%% estimate test error--------------------------------------
num_catch=[];
size_test=1000;
ind=grap(:,end).*(1:m1)';
for i=1:size_test
    X=funNormalize(test(i,:));
    X(isnan(X))=0;
    d_input=data_train_Y_preprocessed(i,:)';
    [catch_X]=forward_propV3(catch_para_layers,grap,X,U);
    Y=catch_X(:,end);
    Y2=Y(ind(ind>0));
    Y2=Y2(1:10);
    num_max=find(Y2==max(Y2))-1;
    num_catch(i)=num_max(1);%record the num
end
num_catch2=data_test_Y_orig(1:size_test)';
e_tem=num_catch-num_catch2;
accu=sum(e_tem==0)/size_test

%% matrix back_propogation
batch_size=10;
epoch=1;
size_train=60000;
size_test=1000;
% t_end=epoch*size_train/batch_size;
W2=rand(m1,m2)-0.5;
catch_cost=[];
% a=10^(-1/t_end);
a=0.9996;
learning_rate=0.1;
catch_cost_test=[];
for i=1:t_end
    index=randperm(batch_size)+batch_size*(i-1);
    train_temX=train(index,:);
    train_temY=data_train_Y_preprocessed(index,:);
    sum_delt_U=0;
    for j=1:batch_size
    X=train_temX(j,:);
    d=train_temY(j,:)';
    v2=(X*W2)';
    y2=sigmoid(v2);
    delt_U=X'*(-2*(d-y2).*(y2.*(1-y2)))';
    sum_delt_U=sum_delt_U+delt_U;
    end
    
    cost=sum((d-y2).^2);
    catch_cost=[catch_cost,cost];
    learning_rate=learning_rate*a;
    W2=W2-learning_rate*sum_delt_U/batch_size;
    %----early stop------
%     if cost<0.1
%         break
%     end
    if rem(i,100)==0
        sum_cost_test=0;
        for k=1:size_test
            X=testset(k,1:m1);
            d=testset(k,m1+1:end)';
            Y=sigmoid(X*W2)';            
            cost_test=sum((d-Y).^2);
            sum_cost_test=sum_cost_test+cost_test;
        end
    cost_test=sum_cost_test/size_test;
    catch_cost_test=[catch_cost_test,cost_test];
    end
end
figure(),plot(1:1:t_end,catch_cost(1:1:end),'r',1:100:t_end,catch_cost_test,'g');
%estimate test error--------------------------------------
catch_error=[];
for i=1:size_test
    X=testset(i,1:m1);
    d=testset(i,m1+1:end)';
    Y=sigmoid(X*W2)';            
    error=sum((d-Y).^2)/m2;
    catch_error=[catch_error,error];
end
test_error=sum(catch_error/size_test)