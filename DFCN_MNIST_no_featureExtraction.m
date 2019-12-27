% clear
sigmoid = @(x) (1./(1+exp(-x)));
hotone = @(X)bsxfun(@eq, X(:), 1:10);%to onehot
funNormalize = @(x) ( x-min(min(x(:))))/( max(max(x))-min(min(x)) + eps);
%------------------------data preprocess------------------------
size_train=60000;
size_test=1000;
load('data_feature_test.mat');
load('data_feature_train.mat'); 
data_train_Y_orig = h5read('train_dataset.h5','/train_labels');
data_test_Y_orig = h5read('test_dataset.h5','/test_labels');
train=train(:,1:32);
test=test(:,1:32);
data_train_Y_preprocessed=hotone(data_train_Y_orig+1);
data_test_Y_preprocessed=hotone(data_test_Y_orig+1);
data_train_Y_preprocessed(10000,16)=0;
data_test_Y_preprocessed(10000,16)=0;
%------------------------data preprocess------------------------
% size_train=60000;
% size_test=1000;
% data_train_X_orig = h5read('train_dataset.h5','/train_images');
% for i=1:size_train
%     train(:,:,i) = imresize(data_train_X_orig(:,:,i),[32 32],'nearest');
% end
% train=reshape(train(:,:,1:size_train),1024,size_train);
% train=(train');
% 
% data_test_X_orig = h5read('test_dataset.h5','/test_images');
% for i=1:size_test
%     test(:,:,i) = imresize(data_test_X_orig(:,:,i)/255,[32 32],'nearest');
% end
% test=reshape(test(:,:,1:size_test),1024,size_test);
% test=(test');

% data_train_Y_orig = h5read('train_dataset.h5','/train_labels');
% data_test_Y_orig = h5read('test_dataset.h5','/test_labels');
% data_train_Y_preprocessed=hotone(data_train_Y_orig+1);
% data_test_Y_preprocessed=hotone(data_test_Y_orig+1);
% data_train_Y_preprocessed(10000,16)=0;
% data_test_Y_preprocessed(10000,16)=0;

%% ------------------------train------------------------
%here forw_propV3 and back_propV2 are choosen
m1=32;m2=16;
c=[2,2,2,2,1];

batch_size=2;
epoch=1;

catch_cost=[];
catch_cost_test=[];

%----------------these parameters should be closed if starting from last point--------
l=1;
U=(rand(2,2,(m1/2),int8(log2(m1)),l));
% load('U.mat');
t_end=2800;
% t_end=epoch*size_train/batch_size;
learning_rate=0.00008;
a=1;
% a=0.9996;
[catch_para_layers,grap]=get_classes(c,m1);
ind=grap(:,end).*(1:m1)';
sum_delt_U=0;
for i=1:t_end
%     index=floor(size_train*rand(batch_size,1))+1;
    index=[1,2];
%     if rem(i,2)==1
%         index=[1];
%     else
%         index=[2];
%     end
    train_temX=train(index,:);
    train_temY=10*(data_train_Y_preprocessed(index,:));
%     sum_delt_U=0;
%     for j=1:batch_size
%     X=(train_temX(j,:));
%     X(isnan(X))=0;
    
    [Y,sum_delt_U,catch_BN_batch]=train_multilayers(catch_para_layers,grap,U,l,1,train_temX,train_temY,batch_size,m1,catch_BN_batch);
    Y=Y(1,:,1)';
    Y2=Y(ind(ind>0));
    Y2=Y2(1:10);
%     sum_delt_U=sum_delt_U+delt_U_multi;
    %-----end------------
%     end
    d=train_temY(1,:);
    d1=zeros(m1,1);
    d1(ind(ind>0))=d;
    cost=sum((d1-Y).^2);
    catch_cost=[catch_cost,cost];
    learning_rate=learning_rate*a;
    U=U-learning_rate*sum_delt_U;
    %----early stop------
    if (cost<0) || isnan(cost)
        break
    end 
    %cost of test--------
%     if rem(i,10000000)==0
%         sum_cost_test=0;
%         for k=1:size_test
%             X=testset(k,1:m1);
%             d_input=testset(k,m1+1:end)';
%             d=zeros(m1,1);ind=grap(:,end).*(1:m1)';
%             d(ind(ind>0))=d_input;
%             [catch_X]=forward_propV3(catch_para_layers,grap,X,U);
%             Y=catch_X(:,end);
%             cost_test=sum((d-Y).^2);
%             sum_cost_test=sum_cost_test+cost_test;
%         end
%     cost_test=sum_cost_test/size_test;
%     catch_cost_test=[catch_cost_test,cost_test];
%     end
    if rem(i,5)==0
        fprintf('the cost of %d iter is %f\n',i,cost)
        if cost<catch_cost(end-1);
%             save('U.mat','U');
        end
        if rem(i,100)==0
            save('U.mat','U');
        end
    end
end
% figure(),plot(1:1:t_end,catch_cost(1:1:end),'r',1:100:t_end,catch_cost_test,'g');
% save('U2.mat','U');
figure(),plot(catch_cost(1,1:end),'r')
%% estimate test error--------------------------------------
% load('U.mat');
num_catch=[];
size_test=100;
ind=grap(:,end).*(1:m1)';
for i=1:size_test
    X=(test(i,:));
%     X(isnan(X))=0;
%     d_input=data_train_Y_preprocessed(i,:)';
    [Y,delt_U_multi]=train_multilayers(catch_para_layers,grap,X,U,d,l,0);
    Y2=Y(ind(ind>0));
    Y2=Y2(1:10);
    num_max=find(Y2==max(Y2))-1;
    num_catch(i)=num_max(1);%record the num
end
num_catch2=data_test_Y_orig(1:size_test)';
e_tem=num_catch-num_catch2;
accu=sum(e_tem==0)/size_test

%% matrix back_propogation
batch_size=5;
epoch=1;
size_train=50000;
size_test=1000;
t_end=epoch*size_train/batch_size;
W2=rand(m1,10);
catch_cost=[];
% a=10^(-1/t_end);
% a=0.9996;
learning_rate=0.0005;
catch_cost_test=[];
for i=1:t_end
    index=floor(size_train*rand(batch_size))+1;
    train_temX=train(index,:);
    train_temY=data_train_Y_preprocessed(index,:);
    sum_delt_U=0;
    for j=1:batch_size
    X=train_temX(j,:);
    d=10*train_temY(j,1:10)';
%     d=(train_temY(j,:));
    v2=(X*W2)';
    
    v2(v2<0)=0.5*v2(v2<0);%lrelu
    y2=v2;
    
    delt_y=y2;%derivative of lrelu
    delt_y(y2<0)=0.5;
    delt_y(y2>=0)=1;
    delt_U=X'*(-2*(d-y2).*delt_y)';
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
    if rem(i,100000000)==0
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
    if rem(i,100)==0
        fprintf('the cost of %d iter is %f\n',i,cost)
    end
end
figure(),plot(catch_cost(1:1:end));
%estimate test error--------------------------------------
catch_error=[];
for i=1:size_test
    X=(test(i,:));
    d=data_train_Y_preprocessed(i,:)';
    d=d(1:10);
    Y2=sigmoid(X*W2)';            
    num_max=find(Y2==max(Y2))-1;
    num_catch(i)=num_max(1);%record the num
end
num_catch2=data_test_Y_orig(1:size_test)';
e_tem=num_catch-num_catch2;
accu=sum(e_tem==0)/size_test