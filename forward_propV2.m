function [catch_X]=forward_propV2(catch_para_layers,X,U)
%this version uses mask
%output the all the hidden layers
% funNormalize = @(x) ( x-min(min(x(:))))/( max(max(x))-min(min(x)) + eps)-0.5;
sigmoid = @(x) (1./(1+exp(-x)));
X_input=X;
[~,~,tem,~]=size(U);
m=tem*2;
l_max=log2(m);
catch_X=zeros(m,l_max);
% norm_catch=zeros(2,l_max);
% X_tem=zeros(1,2);
for i=1:l_max%layer
    step=0;
%     X_after_pulling=zeros(1,m);
    for class=1:catch_para_layers(i)%number of class
        for j=[1:m/2/(2^(i-1))]+step%number of butters in ever class

            v_tem=[X(j),X(j+2^(l_max-i))]*U(:,:,floor((2+j)/2),i);
            X_tem=(v_tem);
            X_tem=sigmoid(v_tem);
            
            X(j)=X_tem(1);
            X(j+2^(l_max-i))=X_tem(2);
        end

        step=step+2^(l_max-i+1);
    end
    catch_X(:,i)=X';
end
catch_X=[X_input',catch_X];
end