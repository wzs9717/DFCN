function x=lrelu(x)
x(x<0)=0.5*x(x<0);
end