function [y_out,sum_delt_U,catch_BN_batch]=train_multilayers(catch_para_layers,grap,U_multi,l,mode,train_temX,train_temY,batch_size,m1,catch_BN_batch)
%catch_x:l_max+1 by m by batch_size
%catch_BN:mean,var,a,b;size:l_max,m,4,batch_size
%mode==1:back prop;  mode==0:only forward prop
% delt_U_multi=zeros(2,2,(m1/2),int8(log2(m1)),l,batch_size);
% X=X_input;
catch_X_multi=[];
% delt_U_multi=[];
X=train_temX';
catch_v_hat_multi=[];
for i=1:l-1
[catch_X,catch_BN_batch(:,:,:,i),catch_v_hat]=forward_prop(X,U_multi(:,:,:,:,i),batch_size,catch_BN_batch(:,:,:,i));
X=reshape(catch_X(end,:,:),m1,batch_size);
catch_X_multi(:,:,:,i)=catch_X;%l_max+1 by m by batch_size by l
catch_v_hat_multi(:,:,:,i)=catch_v_hat;
end

[catch_X,catch_BN_batch(:,:,:,l),catch_v_hat]=forward_propV3(catch_para_layers,grap,X,U_multi(:,:,:,:,end),batch_size,catch_BN_batch(:,:,:,l));
% catch_BN_batch(:,:,1:2,l)=catch_BN;
y_out=catch_X(end,:,:);
if mode==1
for i=1:batch_size
catch_X(:,:,i)=catch_X(:,:,i).*[ones(m1,1),grap]';
end
[delt_U_multi_batch(:,:,:,:,l,:),e_batch,catch_BN_batch(:,:,:,l)]=back_propV2(catch_X,catch_para_layers,grap,U_multi(:,:,:,:,end),train_temY,batch_size,catch_BN_batch(:,:,:,l),catch_v_hat);

for i=l-1:-1:1
[delt_U_multi_batch(:,:,:,:,i,:),e_batch,catch_BN_batch(:,:,:,i)]=back_prop_e(catch_X_multi(:,:,:,i),U_multi(:,:,:,:,i),e_batch,batch_size,catch_BN_batch(:,:,:,i),catch_v_hat_multi(:,:,:,i));
end

end
sum_delt_U=sum(delt_U_multi_batch,6)/batch_size;
end