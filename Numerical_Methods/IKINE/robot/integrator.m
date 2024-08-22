function f=integrator(dy0,dy1,dy2,dy3,h,i)
%n=size(dy0,1)/2;
if i==1
    f=h*dy3;
elseif (i==2)
    f=h/2.0*(dy2+dy3);
elseif(i==3)
    f=h/3.0*(dy1+4*dy2+dy3);
else
    f=3.0*h/8.0*(dy0+3*dy1+3*dy2+dy3);
end
%f1=f(1:n);
%f2=f(n+1:2*n);
