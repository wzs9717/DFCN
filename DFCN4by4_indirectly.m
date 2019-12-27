%------------------------------------
% 4by4 indirectly
% --------------------------------
% funNormalize = @(x) ( x-min(min(x(:))))/( max(max(x))-min(min(x)) + eps)-0.5;
sigmoid = @(x) (1./(1+exp(-x)));
%% preprocess------------------------
%here we get a matrix U by fit the data (x,y) produced by W with FFT frame.
m=4;
U0=rand(2,2,(m/2),int8(log2(m)));
U=U0;
% W2=[U(1,1,1,2)*U(1,1,1,1),U(2,1,1,2)*U(1,1,2,1),U(1,1,1,2)*U(2,1,1,1),U(2,1,1,2)*U(2,1,2,1);
%         U(1,2,1,2)*U(1,1,1,1),U(2,2,1,2)*U(1,1,2,1),U(1,2,1,2)*U(2,1,1,1),U(2,2,1,2)*U(2,1,2,1);
%         U(1,1,2,2)*U(1,2,1,1),U(2,1,2,2)*U(1,2,2,1),U(1,1,2,2)*U(2,2,1,1),U(2,1,2,2)*U(2,2,2,1);
%         U(1,2,2,2)*U(1,2,1,1),U(2,2,2,2)*U(1,2,2,1),U(1,2,2,2)*U(2,2,1,1),U(2,2,2,2)*U(2,2,2,1)]';

%FP--------------------------------------
% X=rand(1,m);
% [catch_X]=forward_prop(X,U)
% Y=catch_X(:,end)
% W_equl=[U(1,1,1,2)*U(1,1,1,1),U(2,1,1,2)*U(1,1,2,1),U(1,1,1,2)*U(2,1,1,1),U(2,1,1,2)*U(2,1,2,1);
%         U(1,2,1,2)*U(1,1,1,1),U(2,2,1,2)*U(1,1,2,1),U(1,2,1,2)*U(2,1,1,1),U(2,2,1,2)*U(2,1,2,1);
%         U(1,1,2,2)*U(1,2,1,1),U(2,1,2,2)*U(1,2,2,1),U(1,1,2,2)*U(2,2,1,1),U(2,1,2,2)*U(2,2,2,1);
%         U(1,2,2,2)*U(1,2,1,1),U(2,2,2,2)*U(1,2,2,1),U(1,2,2,2)*U(2,2,1,1),U(2,2,2,2)*U(2,2,2,1)]';%the equal matrix of U
% D=sigmoid(X*W_equl)
% y1=U(1,1,1,2)*(U(1,1,1,1)*X(1)+U(2,1,1,1)*X(3))+U(2,1,1,2)*(U(1,1,2,1)*X(2)+U(2,1,2,1)*X(4))

%BP-------------------------------------------------------------------------\
para_complex=32;
batch_size=10;
epoch=5;
size_train=10000;
size_test=100;
t_end=epoch*size_train/batch_size;

% trainsets---------------------------------------
W=(rand(para_complex*m,para_complex*m))-0.5;
for i=1:size_train
X=(rand(1,para_complex*m))*10;
d=sigmoid(X*W)';
trainsetX(i,:)=X(1:end/para_complex);
trainsetY(i,:)=d(1:end/para_complex)';
end
trainset=[trainsetX,trainsetY];
%testsets---------------------------------------

for i=1:size_test
X=(rand(1,para_complex*m))*10;
d=sigmoid(X*W)';
testsetX(i,:)=X(1:end/para_complex);
testsetY(i,:)=d(1:end/para_complex)';
end
testset=[testsetX,testsetY];
%% train-----------------------------------
catch_cost=[];
catch_cost_test=[];
a=10^(-1/t_end);
learning_rate=0.1;
for i=1:t_end
%     if 0
%     X=rand(1,m);
%     endhh
%     d=(X*W)';
    train_tem=trainset(randperm(batch_size),:);
    sum_delt_U=0;
    for j=1:batch_size
    X=train_tem(j,1:end/2);
    d=train_tem(j,end/2+1:end)';
    [catch_X,norm_catch]=forward_prop(X,U);
    Y=catch_X(:,end);
    delt_U=back_prop([X',catch_X],U,d);
    sum_delt_U=sum_delt_U+delt_U;
    %----W_res2---------
%     Y2=X*W_res2;
%     delt_W_res2=-2*(d-Y2)'*X;
%     sum_delt_U=sum_delt_U+delt_U;
    %-----end------------
    end
    
    cost=sum((d-Y).^2);
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
            X=testset(k,1:end/2);
            d=testset(k,end/2+1:end)';
            [catch_X]=forward_prop(X,U);
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
W_equl=[U(1,1,1,2)*U(1,1,1,1),U(2,1,1,2)*U(1,1,2,1),U(1,1,1,2)*U(2,1,1,1),U(2,1,1,2)*U(2,1,2,1);
        U(1,2,1,2)*U(1,1,1,1),U(2,2,1,2)*U(1,1,2,1),U(1,2,1,2)*U(2,1,1,1),U(2,2,1,2)*U(2,1,2,1);
        U(1,1,2,2)*U(1,2,1,1),U(2,1,2,2)*U(1,2,2,1),U(1,1,2,2)*U(2,2,1,1),U(2,1,2,2)*U(2,2,2,1);
        U(1,2,2,2)*U(1,2,1,1),U(2,2,2,2)*U(1,2,2,1),U(1,2,2,2)*U(2,2,1,1),U(2,2,2,2)*U(2,2,2,1)]';
    
%estimate test error--------------------------------------
catch_error=[];
for i=1:size_test
    X=testset(i,1:end/2);
    d=testset(i,end/2+1:end)';
    [catch_X]=forward_prop(X,U);
    Y=catch_X(:,end);
    error=sum(abs(d-Y))/m;
    catch_error=[catch_error,error];
end
test_error=sum(catch_error/size_test)

%% matrix back_propogation
batch_size=10;
epoch=5;
size_train=10000;
size_test=100;
t_end=epoch*size_train/batch_size;
W2=rand(m,m)-0.5;
catch_cost=[];
a=10^(-1/t_end);
learning_rate=0.1;
catch_cost_test=[];
for i=1:t_end
%     if 0
%     X=rand(1,m);
%     endhh
%     d=(X*W)';
    train_tem=trainset(randperm(batch_size),:);
    sum_delt_U=0;
    for j=1:batch_size
    X=train_tem(j,1:end/2);
    d=train_tem(j,end/2+1:end)';
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
            X=testset(k,1:end/2);
            d=testset(k,end/2+1:end)';
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
    X=testset(i,1:end/2);
    d=testset(i,end/2+1:end)';
    Y=sigmoid(X*W2)';            
    error=sum(abs(d-Y))/m;
    catch_error=[catch_error,error];
end
test_error=sum(catch_error/size_test)