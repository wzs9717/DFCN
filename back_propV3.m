function [delt_U]=back_propV3(catch_X,catch_para_layers,grap,U,d)
%here X_catch include X, this version(2.0) compute delt_U with unfixed
%butterflies, so the number of channels(c) should be given
c=catch_para_layers(4,1:end-1);
[~,~,tem,~]=size(U);
m=tem*2;
l_max=log2(m);
% d=norm5(d);
delt_U=zeros(size(U));
Y=catch_X(:,end);
for i=l_max:-1:1
    count=1;
    step_butt=2.^(l_max-i);
%     m_curr=catch_para_layers(3,i);
    class=catch_para_layers(1,i);
    m_class=catch_para_layers(2,i);
    step_class=m_class;
    e_layer=[];
    if i==l_max
        e_tem=-2*(d-Y).*(Y.*(1-Y));
        e=reshape(e_tem(1:catch_para_layers(3,i+1)),c(i),catch_para_layers(3,i+1)/c(i));
        clear e_tem;
        for j=1:class
            for k=1:m_class/2
                v=catch_X(:,i);
                delt_U(:,1:c(i),k+(j-1)*m_class/2,i)=[v(k+(j-1)*step_class);v(k+(j-1)*step_class+step_butt)]*e(:,count)';
                count=count+1;
            end
        end
    else
        for j=1:class
            for k=1:m_class/2
                v=catch_X(1:catch_para_layers(3,i),i);
                y=catch_X(1:catch_para_layers(3,i+1),i+1);
                y_tem=[y(k+(j-1)*step_class),y(k+(j-1)*step_class+step_butt)];
                e_tem(count,:)=e(count,:)*U(:,1:c(i),k+(j-1)*m_class/2,i+1)'.*(y_tem.*(1-y_tem));
                delt_U(:,1:c(i),k+(j-1)*m_class/2,i)=[v(k+(j-1)*step_class);v(k+(j-1)*step_class+step_butt)]*e_tem(count,:);
                count=count+1;
            end
            e_layer=[e_layer;e_tem(:)];
            clear e_tem;
        end
        e=reshape(e_layer(:),c(i),catch_para_layers(3,i+1)/c(i));
    end
end
end