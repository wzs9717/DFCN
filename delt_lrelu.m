function y=delt_lrelu(x)
y=x;
y(x<0)=0.5;
y(x>=0)=1;
end