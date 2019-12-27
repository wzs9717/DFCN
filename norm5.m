function [y,maxx,minn]=norm5(x)
minn=min(x(:));
maxx=max(x(:));
y=(x-minn)./(maxx-minn)-0.5;
end