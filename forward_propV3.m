function [catch_X,catch_BN,catch_v_hat]=forward_propV3(catch_para_layers,grap,X,U,batch_size,catch_BN)
%this version can be used for any size channels
%output the all the hidden layers
% funNormalize = @(x) ( x-min(min(x(:))))/( max(max(x))-min(min(x)) + eps)-0.5;
% lrelu= @(x) (x(x<0)=0.5*x(x<0));
% sigmoid = @(x) (1./(1+exp(-x)));
X_input=X;
[~,~,tem,~]=size(U);
m_input=tem*2;
l_max=log2(m_input);
% catch_BN=zeros(l_max,m,2);
catch_X=zeros(l_max+1,m_input,batch_size);%here m_output shoule be less or equal than m_input
catch_X(1,:,:)=X;
c=catch_para_layers(4,1:end-1);
catch_WX=zeros(m,batch_size);
v=zeros(m,1);
catch_v_hat=zeros(l_max,m,batch_size);
for i=1:l_max
    for o=1:batch_size
    if i==1
        X=X_input(:,o);
    else
        X=catch_X(i,:,o);
    end
    count=1;
    m_curr=catch_para_layers(3,i);
    class=catch_para_layers(1,i);
    step_butt=2.^(l_max-i);
    m_class=catch_para_layers(2,i);
    step_class=m_class;
    v_layer=[];
    for j=1:2^(i-1)
        for k=1:m_class/2
            if grap(k+(j-1)*step_class,i)==0
                v_tem(k,:)=zeros(1,c(i));
                count=count+1;
                continue
            end
            v_tem=[X(k+(j-1)*step_class),X(k+(j-1)*step_class+step_butt)]*U(:,1:c(i),count,i);
            count=count+1;
%             v(v<0)=0.5*v(v<0);%lrelu
%             X_tem(k,:)=(v);
            
        end
        v_tem2=zeros(m_class/2,2);%reshape v_tem to standard shape 2by2
        v_tem2(:,1:c(i))=v_tem;
        v_layer=[v_layer;v_tem2(:)];
        clear X_tem;
    end
%     catch_X(i+1,1:length(v_layer),o)=v_layer;
    v=zeros(m,1);
    v(1:length(v_layer))=v_layer;
%     if size(X)<m_input                    I am not sure if to drop this 
%         X(m_input)=0;
%     end
    catch_WX(:,o)=v;
    end
catch_BN(i,:,1)=mean(catch_WX,2);
catch_BN(i,:,2)=var(catch_WX,2);
v_hat=(catch_WX-catch_BN(i,:,1)*ones(1,m))./((catch_BN(i,:,2).^0.5)*ones(1,m)+eps);
% v_hat=catch_WX.*((catch_BN(i,:,3)./(catch_BN(i,:,2).^0.5+eps))*ones(1,m))+(catch_BN(i,:,4)-catch_BN(i,:,1)./(catch_BN(i,:,2).^0.5+eps))*ones(1,m);
catch_v_hat(i,:,:)=v_hat;
u=(catch_BN(i,:,3)*ones(1,m)).*v_hat+catch_BN(i,:,4)*ones(1,m);
u(u<0)=0.5*u(u<0);%lrelu
catch_X(i+1,:,:)=u;
    
% catch_BN(i,:,1)=mean(catch_WX,2);
% catch_BN(i,:,2)=var(catch_WX,2);
% v=catch_WX.*((catch_BN(i,:,3)./(catch_BN(i,:,2).^0.5+eps))*ones(1,m))+(catch_BN(i,:,4)-catch_BN(i,:,1)./(catch_BN(i,:,2).^0.5+eps))*ones(1,m);
% v(v<0)=0.5*v(v<0);%lrelu
% % catch_X(i+1,:,:)=v;
% catch_X(i+1,:,:)=v;
end

end