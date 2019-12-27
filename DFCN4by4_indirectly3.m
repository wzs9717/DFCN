%% preprocess------------------------
sigmoid = @(x) (1./(1+exp(-x)));
m1=1024;m2=16;
c=[2,2,2,2,1,1,1,1,1,1];
[catch_para_layers,grap]=get_classes(c,m1);
X=rand(1,m1);
d=rand(1,m2)';
%here we get a matrix U by fit the data (x,y) produced by W with FFT frame.
para_complex=1;
batch_size=2;
epoch=2;
size_train=1000;
size_test=100;
t_end=epoch*size_train/batch_size;

% trainsets---------------------------------------
W=(rand(para_complex*m1,para_complex*m2))-0.5;
% for i=1:size_train
% X=(rand(1,para_complex*m1))*10;
% d1=sigmoid(X*W)';
% d=zeros(size(d1));
% d(floor(10*rand(1))+1)=1;
% trainsetX(i,:)=X(1:end/para_complex);
% trainsetY(i,:)=d(1:end/para_complex)';
% end
% trainset=[trainsetX,trainsetY];
for i=1:size_train
X=10*train(i,:);
d=data_train_Y_preprocessed(i,:);
trainsetX(i,:)=X;
trainsetY(i,:)=d;
end
trainset=[trainsetX,trainsetY];
%testsets---------------------------------------

for i=1:size_test
X=(rand(1,para_complex*m1))*10;
d=sigmoid(X*W)';

testsetX(i,:)=X(1:end/para_complex);
testsetY(i,:)=d(1:end/para_complex)';
end
testset=[testsetX,testsetY];
%% train multi layers-----------------------------------
%here forw_propV3 and back_propV2 are choosen
% c=[2,2,1];
% m1=8;m2=4;
[catch_para_layers,grap]=get_classes(c,m1);
l=1;
U0=rand(2,2,(m1/2),int8(log2(m1)),l);
U=U0;
batch_size=10;
epoch=1;
size_train=10000;
size_test=100;
t_end=epoch*size_train/batch_size;

catch_cost=[];
catch_cost_test=[];
a=10^(-1/t_end);
learning_rate=0.001;
for i=1:t_end
    train_tem=trainset(randperm(batch_size)+batch_size*(i-1),:);
    sum_delt_U=0;
    for j=1:batch_size
    X=train_tem(j,1:m1);
    d=train_tem(j,m1+1:end)';
    [Y,delt_U_multi]=train_multilayers(catch_para_layers,grap,X,U,d,l,1);
    sum_delt_U=sum_delt_U+delt_U_multi;
    end
    
    d1=zeros(m1,1);ind=grap(:,end).*(1:m1)';
    d1(ind(ind>0))=d;
    cost=sum((d1-Y).^2);
    catch_cost=[catch_cost,cost];
    learning_rate=learning_rate*a;
    U=U-learning_rate*sum_delt_U/batch_size;
    %----early stop------
%     if cost<0.1
%         break
%     end
    %cost of test--------
    if rem(i,100)==0
        sum_cost_test=0;
        for k=1:size_test
            X=testset(k,1:m1);
            d_input=testset(k,m1+1:end)';
            d=zeros(m1,1);ind=grap(:,end).*(1:m1)';
            d(ind(ind>0))=d_input;
            [Y,delt_U_multi]=train_multilayers(catch_para_layers,grap,X,U,d,l,0);
            cost_test=sum((d-Y).^2);
            sum_cost_test=sum_cost_test+cost_test;
        end
    cost_test=sum_cost_test/size_test;
    catch_cost_test=[catch_cost_test,cost_test];
    end
    if rem(i,5)==0
        fprintf('the cost of %d iter is %f\n',i,cost)
    end
end
    
%test-------------------------------------------------
% figure(),plot(1:1:t_end,catch_cost(1:1:end),'r',1:100:t_end,catch_cost_test,'g');
plot(catch_cost(1:1:end));
%estimate test error--------------------------------------
catch_error=[];
for i=1:size_test
    X=testset(i,1:m1);
    d_input=testset(i,m1+1:end)';
    d=zeros(m1,1);ind=grap(:,end).*(1:m1)';
    d(ind(ind>0))=d_input;
    [Y,delt_U_multi]=train_multilayers(catch_para_layers,grap,X,U,d,l,0);
%     [catch_X]=forward_propV3(catch_para_layers,grap,X,U);
    error=sum((d-Y).^2)/m2;
    catch_error=[catch_error,error];
end
test_error=sum(catch_error/size_test)
