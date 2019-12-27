function [delt_U_batch,e_batch,delt_ab]=back_prop_e(catch_X_batch,U,e_batch,batch_size,catch_BN,catch_v_hat)
%here X_catch include X, this version(1.0) compute delt_U with fixed 2by2 butterflies
%this version is applied in multi-layers back prop, so d is replaced with e
[~,~,tem,~]=size(U);
m=tem*2;
l_max=log2(m);
% e_batch=zeros(m,batch_size);
delt_U=zeros(size(U));
tem=size(U);
delt_U_batch=zeros(tem(1),tem(2),tem(3),tem(4),batch_size);
% Y=catch_X(:,end);
for i=l_max:-1:1%layer
    for o=1:batch_size
    catch_X=catch_X_batch(:,:,o)';
    e=e_batch(:,o);
    count=1;
    step=0;
    if i==l_max
        %output layer
        for class=1:2^(i-1)%number of class
            for j=[1:m/2/(2^(i-1))]+step%number of butters in ever class
                y=[catch_X(j,i+1),catch_X(j+2^(l_max-i),i+1)];
                delt_y=y;%derivative of lrelu
                delt_y(y<0)=0.5;
                delt_y(y>=0)=1;
                e_tem=[e(j),e(j+2^(l_max-i))].*delt_y;
                e(j)=e_tem(1);
                e(j+2^(l_max-i))=e_tem(2);
                
                v=catch_X(:,i);
                delt_U(:,:,count,i)=[v(j);v(j+2^(l_max-i))]*(e_tem);
                count=count+1;
            end
            step=step+2^(l_max-i+1);
        end 
    else
        %hidden layer
        for class=1:2^(i-1)%number of class
            for j=[1:m/2/(2^(i-1))]+step%number of butters in ever class
                y=[catch_X(j,i+1),catch_X(j+2^(l_max-i),i+1)];
                delt_y=y;%derivative of lrelu
                delt_y(y<0)=0.5;
                delt_y(y>=0)=1;
                e_tem=[e(j),e(j+2^(l_max-i))]*U(:,:,count,i+1)'.*delt_y;
                e(j)=e_tem(1);
                e(j+2^(l_max-i))=e_tem(2);       
                v=catch_X(:,i);
                delt_U(:,:,count,i)=[v(j);v(j+2^(l_max-i))]*(e_tem);
                count=count+1;
            end
            step=step+2^(l_max-i+1);
        end
    end
    delt_U_batch(:,:,:,:,o)=delt_U;
    e_batch(:,o)=e;
    end
end
%to pass to anothoer layer
for o=1:batch_size
    e=e_batch(:,o);
for i=1:m/2
    e_tem=[e(i),e(i+2^(l_max-1))]*U(:,:,i,1)';
    e(i)=e_tem(1);
    e(i+2^(l_max-1))=e_tem(2); 
end
    e_batch(:,o)=e;
end

end