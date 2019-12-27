function [catch_para_layers,grap]=get_classes(c,m)
%get the parameters:
%     class;
%     m_class;
%     m_curr;
%      c
%and computing graph
l_max=log2(m);
m_curr=m;
class=1;
catch_para_layers=[];
grap=zeros(m,l_max);
X=ones(1,m);
for i=1:l_max
    step=0;
    count=1;
        for classes=1:2^(i-1)%number of class
        for j=[1:m/2/(2^(i-1))]+step%number of butters in ever class
        if c(i)==1
            X_tem=[X(j),X(j+2^(l_max-i))]*[1,0;1,0];
            X(j)=X_tem(1);
            X(j+2^(l_max-i))=X_tem(2);
        else
            X_tem=[X(j),X(j+2^(l_max-i))]*[1,1;1,1];
            X(j)=X_tem(1);
            X(j+2^(l_max-i))=X_tem(2);
        end
        end
        step=step+2^(l_max-i+1);
        end
        grap(:,i)=X';
        count=count+1;
    m_curr=m_curr*c(i)/2;
    class=class*c(i);
    m_class=m_curr/class;
    catch_para_layers(1,i)=class;
    catch_para_layers(2,i)=m_class;
    catch_para_layers(3,i)=m_curr;
end
catch_para_layers=[[1;m;m],catch_para_layers;c,0];
grap=(grap>0);
end