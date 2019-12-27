
%train a DFCN for MNIST datasets
% c=[2,2,2];
% m1=8;m2=8;
% [catch_para_layers,grap]=get_classes(c,m1);
% X=rand(1,m1);
% d=rand(1,m2)';
% U0=rand(2,2,(m1/2),int8(log2(m1)));
% U=U0;
% [catch_X]=forward_propV3(catch_para_layers,grap,X,U)
% delt_U1=back_propV2(catch_X,catch_para_layers,grap,U,d);
% 
% [catch_X]=forward_prop(X,U)
% Y=catch_X(:,end);
% delt_U2=back_prop([X',catch_X],U,d)
% sum(abs(delt_U1-delt_U2))

%% preprocess------------------------
sigmoid = @(x) (1./(1+exp(-x)));
c=[2,2,1];
m1=8;m2=4;
[catch_para_layers,grap]=get_classes(c,m1);
X=rand(1,m1);
d=rand(1,m2)';
%here we get a matrix U by fit the data (x,y) produced by W with FFT frame.
para_complex=16;
batch_size=10;
epoch=5;
size_train=10000;
size_test=100;
t_end=epoch*size_train/batch_size;

% trainsets---------------------------------------
W=(rand(para_complex*m1,para_complex*m2))-0.5;
for i=1:size_train
X=(rand(1,para_complex*m1))*10;
d=sigmoid(X*W)';
trainsetX(i,:)=X(1:end/para_complex);
trainsetY(i,:)=d(1:end/para_complex)';
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
%% train-----------------------------------
%here forw_propV3 and back_propV2 are choosen
U0=rand(2,2,(m1/2),int8(log2(m1)));
U=U0;
batch_size=10;
epoch=1;
size_train=10000;
size_test=100;
t_end=epoch*size_train/batch_size;

catch_cost=[];
catch_cost_test=[];
a=10^(-1/t_end);
learning_rate=0.01;
for i=1:t_end
    train_tem=trainset(randperm(batch_size),:);
    sum_delt_U=0;
    for j=1:batch_size
    X=train_tem(j,1:m1);
    d=train_tem(j,m1+1:end)';
    [catch_X]=forward_propV3(catch_para_layers,grap,X,U);
    Y=catch_X(:,end);
    delt_U=back_propV2(catch_X,catch_para_layers,grap,U,d);
    sum_delt_U=sum_delt_U+delt_U;
    %----W_res2---------
%     Y2=X*W_res2;
%     delt_W_res2=-2*(d-Y2)'*X;
%     sum_delt_U=sum_delt_U+delt_U;
    %-----end------------
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
            [catch_X]=forward_propV3(catch_para_layers,grap,X,U);
            Y=catch_X(:,end);
            cost_test=sum((d-Y).^2);
            sum_cost_test=sum_cost_test+cost_test;
        end
    cost_test=sum_cost_test/size_test;
    catch_cost_test=[catch_cost_test,cost_test];
    end
end
    
%test-------------------------------------------------
figure(),plot(1:1:t_end,catch_cost(1:1:end),'r',1:100:t_end,catch_cost_test,'g');
% figure(),plot(1:1:t_end,catch_cost(1:1:end))
% W_equl=[U(1,1,1,2)*U(1,1,1,1),U(2,1,1,2)*U(1,1,2,1),U(1,1,1,2)*U(2,1,1,1),U(2,1,1,2)*U(2,1,2,1);
%         U(1,2,1,2)*U(1,1,1,1),U(2,2,1,2)*U(1,1,2,1),U(1,2,1,2)*U(2,1,1,1),U(2,2,1,2)*U(2,1,2,1);
%         U(1,1,2,2)*U(1,2,1,1),U(2,1,2,2)*U(1,2,2,1),U(1,1,2,2)*U(2,2,1,1),U(2,1,2,2)*U(2,2,2,1);
%         U(1,2,2,2)*U(1,2,1,1),U(2,2,2,2)*U(1,2,2,1),U(1,2,2,2)*U(2,2,1,1),U(2,2,2,2)*U(2,2,2,1)]';
%     
%estimate test error--------------------------------------
catch_error=[];
for i=1:size_test
    X=testset(i,1:m1);
    d_input=testset(i,m1+1:end)';
    d=zeros(m1,1);ind=grap(:,end).*(1:m1)';
    d(ind(ind>0))=d_input;
    [catch_X]=forward_propV3(catch_para_layers,grap,X,U);
    Y=catch_X(:,end);
    error=sum((d-Y).^2)/m2;
    catch_error=[catch_error,error];
end
test_error=sum(catch_error/size_test)

%% matrix back_propogation
batch_size=10;
epoch=5;
size_train=10000;
size_test=100;
t_end=epoch*size_train/batch_size;
W2=rand(m1,m2)-0.5;
catch_cost=[];
a=10^(-1/t_end);
% learning_rate=0.1;
catch_cost_test=[];
for i=1:t_end
    train_tem=trainset(randperm(batch_size),:);
    sum_delt_U=0;
    for j=1:batch_size
    X=train_tem(j,1:m1);
    d=train_tem(j,m1+1:end)';
    v2=(X*W2)';
    y2=sigmoid(v2);
    delt_U=X'*(-2*(d-y2).*(y2.*(1-y2)))';
    sum_delt_U=sum_delt_U+delt_U;
    %----W_res2---------
%     Y2=X*W_res2;
%     delt_W_res2=-2*(d-Y2)'*X;
%     sum_delt_U=sum_delt_U+delt_U;
    %-----end------------
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

