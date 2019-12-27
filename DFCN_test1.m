% % a=[0,0,1,2;3,5,7,11;0,0,13,17;19,23,29,43;];
% tem=0;
% for i=1:10000
% a=rand(4,4);
% b=a;
% b(1,1)=b(1,3)/b(2,3)*b(2,1);
% b(1,2)=b(1,4)/b(2,4)*b(2,2);
% b(3,1)=b(3,3)/b(4,3)*b(4,1);
% b(3,2)=b(3,4)/b(4,4)*b(4,2);
% tem=tem+svd(b);
% end
% tem/10000
%% 2by2--------------------------------
catch_E=[];
for j=1:10
W=rand(2,2);
W_res=rand(2,2);
catch_J=[];
for i=1:100
    a=W_res(1,1);b=W_res(1,2);c=W_res(2,1);
    a1=W(1,1);b1=W(1,2);c1=W(2,1);d1=W(2,2);
    J=[2*(a-a1-(c/a*b-d1)*c*b*a^(-2)),2*(b-b1+(c/a*b-d1)*c/a);2*(c-d1+(b/a*c-d1)*b/a),0];
    W_res=W_res-0.08*J;
    W_res(2,2)=b*c/a;
%     catch_J=[catch_J;J(1,1)];
end
E=[];
for i=1:100
    X=rand(2,2);
    error=sum(sum(abs(X*W-X*W_res)))/2;
    E=[E;error];
end
E=sum(E)/100;
catch_E=[catch_E;E];
end
% plot(1:100,catch_J);
sum(catch_E)/10

%% ------4by4 directly--------------------------------
%here we get a matrix W_restricted by appromix it to W directly
catch_E=[];
t1=1;
t_iter=500;
t_error=100;
sum_cost=0;
sum_singu_matU=0;
sum_singu_matW=0;
for j=1:t1
W=rand(4,4)+0.5;
W_res=rand(4,4)+0.5;
catch_J=[];
for i=1:t_iter
    b=[W_res(1,3),W_res(1,4);W_res(3,3),W_res(3,4)];
    c=[W_res(2,1),W_res(2,2);W_res(4,1),W_res(4,2)];
    a=[W_res(2,3),W_res(2,4);W_res(4,3),W_res(4,4)];
    gamma=[W(1,1),W(1,2);W(3,1),W(3,2)];
    delt_base=b.*c./a-gamma;
    delt_b=delt_base.*c./a;
    delt_c=delt_base.*b./a;
    delt_a=-delt_base.*b.*c.*(a.^(-2));
    
    J2=[0,0,delt_b(1,:);delt_c(1,:),delt_a(1,:);0,0,delt_b(2,:);delt_c(2,:),delt_a(2,:)];
    J1=W_res-W;
    J=J1+J2;
    W_res=W_res-0.02*J;
    cost=sum(sum((W_res-W).^2));
    b=[W_res(1,3),W_res(1,4);W_res(3,3),W_res(3,4)];
    c=[W_res(2,1),W_res(2,2);W_res(4,1),W_res(4,2)];
    a=[W_res(2,3),W_res(2,4);W_res(4,3),W_res(4,4)];
    U_tem=b.*c./a;
    W_res(1,1:2)=U_tem(1,1:2);
    W_res(3,1:2)=U_tem(2,1:2);
    catch_J=[catch_J;cost];
end
sum_cost=sum_cost+cost;
sum_singu_matU=sum_singu_matU+svd(W_res);
sum_singu_matW=sum_singu_matW+svd(W);
E=[];
for i=1:t_error
    X=rand(1,4);
    error=(sum(sum(abs(X*W-X*W_res))))/4;
    E=[E;error];
end 
E=sum(E)/t_error;
catch_E=[catch_E;E];
end
plot(1:t_iter,catch_J);
average_error=sum(catch_E)/t1%average erroe
average_cost=sum_cost/t1%average cost
average_singular_U=sum_singu_matU/t1%average singular
average_singular_W=sum_singu_matW/t1
ratial_of_condition_number=(sum_singu_matU(1)/sum_singu_matU(end))/(sum_singu_matW(1)/sum_singu_matW(end))%the ratial of condition number


%% ------4by4 directly--------------------------------
d=(X*W)';
W_res2=rand(4,4)*0.5+W;
catch_cost=[];
for i=1:1000              
    Y2=(X*W_res2)';
    delt_W_res2=-2*(d-Y2)*X;
    delt_W_res2(1,2).*delt_W_res2(2,1)./delt_W_res2(1,1)-delt_W_res2(2,2)
    W_res2=W_res2-0.01*delt_W_res2;
    cost=sum((d-Y2).^2);s
    catch_cost=[catch_cost,cost];
end
figure(),plot(1:1000,catch_cost);
