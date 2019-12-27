function [catch_X,catch_BN,catch_v_hat]=forward_prop(train_temX,U,batch_size,catch_BN)
%output the all the hidden layers

% funNormalize = @(x) ( x-min(min(x(:))))/( max(max(x))-min(min(x)) + eps)-0.5;
% X=X_input;

[~,~,tem,~]=size(U);
m=tem*2;
l_max=log2(m);
% catch_BN=zeros(l_max,m,2);
catch_X=zeros(l_max+1,m,batch_size);
catch_X(1,:,:)=(train_temX);
catch_v_hat=zeros(l_max,m,batch_size);
catch_WX=zeros(m,batch_size);
v=zeros(m,1);
for i=1:l_max%layer

for k=1:batch_size
    step=0;
    count=1;
    if i==1
        X=train_temX(:,k);
    else
        X=catch_X(i,:,k);
    end
    for class=1:2^(i-1)%number of class
        for j=[1:m/2/(2^(i-1))]+step%number of butters in ever class
            v_tem=[X(j),X(j+2^(l_max-i))]*U(:,:,count,i);
            count=count+1;
            v(j)=v_tem(1);
            v(j+2^(l_max-i))=v_tem(2);
%             v(v<0)=0.5*v(v<0);%lrelu
%             X_tem=(v);
%             
%             X(j)=X_tem(1);
%             X(j+2^(l_max-i))=X_tem(2);
        end
        step=step+2^(l_max-i+1);
    end
%     catch_X(i+1,:,k)=X;
    catch_WX(:,k)=v;
end
catch_BN(i,:,1)=mean(catch_WX,2);
catch_BN(i,:,2)=var(catch_WX,2);
v_hat=(catch_WX-catch_BN(i,:,1)*ones(1,m))./((catch_BN(i,:,2).^0.5)*ones(1,m)+eps);
% v_hat=catch_WX.*((catch_BN(i,:,3)./(catch_BN(i,:,2).^0.5+eps))*ones(1,m))+(catch_BN(i,:,4)-catch_BN(i,:,1)./(catch_BN(i,:,2).^0.5+eps))*ones(1,m);
catch_v_hat(i,:,:)=v_hat;
u=(catch_BN(i,:,3)*ones(1,m)).*v_hat+catch_BN(i,:,4)*ones(1,m);
u(u<0)=0.5*u(u<0);%lrelu
catch_X(i+1,:,:)=u;
end
% catch_BN_batch(:,:,1:2,i)=catch_BN;
end