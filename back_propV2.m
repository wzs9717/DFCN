function [delt_U_batch,e_batch,catch_BN,delt_ab]=back_propV2(catch_X_batch,catch_para_layers,grap,U,train_temY,batch_size,catch_BN,catch_v_hat)
%here X_catch include X, this version(1.0) compute delt_U with fixed 2by2 butterflies
[~,~,tem,~]=size(U);
m=tem*2;
% c=catch_para_layers(4,1:end-1);
l_max=log2(m);
e_batch=zeros(m,batch_size);
delt_ab=zeros(l_max,m,2);
% d=d'*([1:length(d)]'*grap(:,end)');
delt_U=zeros(size(U));
tem=size(U);
delt_U_batch=zeros(tem(1),tem(2),tem(3),tem(4),batch_size);

for i=l_max:-1:1%layer
    for o=1:batch_size
    catch_X=catch_X_batch(:,:,o)';%l_max+1 by m
    
    
%     step_butt=2.^(l_max-i);
%     classes=catch_para_layers(1,i);
%     m_class=catch_para_layers(2,i);
%     step_class=m_class;
    count=1;
    step=0;
%     e_layer=[];
    if i==l_max
        Y=catch_X(:,end);
        d_input=train_temY(o,:);
        d=zeros(m,1);ind=grap(:,end).*(1:m)';%size convert
        d(ind(ind>0))=d_input;
        %output layer
        for class=1:2^(i-1)%number of class
            for j=[1:m/2/(2^(i-1))]+step%number of butters in ever class
                if grap(j,i)==0
                    e_tem=zeros(1,2);
                    count=count+1;
                    e(j)=e_tem(1);
                    e(j+2^(l_max-i))=e_tem(2);
                    continue
                end
                y=[catch_X(j,i+1),catch_X(j+2^(l_max-i),i+1)];
                
                delt_y=y;%derivative of lrelu
                delt_y(y<0)=0.5;
                delt_y(y>=0)=1;
                
                e_tem=-2*([d(j),d(j+2^(l_max-i))]-[Y(j),Y(j+2^(l_max-i))]).*delt_y; 
                
                e(j)=e_tem(1);
                e(j+2^(l_max-i))=e_tem(2);
                
                delt_v_hat=[e_tem(1),e_tem(2)].*[catch_BN(i,j,3),catch_BN(i,(j+2^(l_max-i)),3)];
                %------这里需要再加一层，因为的计算完所有delt_v_hat才能继续往前传递---------

                
                v=catch_X(:,i);
                delt_U(:,:,count,i)=[v(j);v(j+2^(l_max-i))]*(e_tem);
                count=count+1;
            end
            step=step+2^(l_max-i+1);
        end 
    else
        %hidden layer
        e=e_batch(:,o);
        for class=1:2^(i-1)%number of class
            for j=[1:m/2/(2^(i-1))]+step%number of butters in ever class
                if grap(j,i)==0
                    e_tem=zeros(1,2);
                    count=count+1;
                    e(j)=e_tem(1);
                    e(j+2^(l_max-i))=e_tem(2);
                    continue
                end
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
    delt_ab(i,:,2)=sum(e_batch,2)./batch_size;
    delt_ab(i,:,1)=sum(e_batch.*catch_v_hat,2)./batch_size;
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