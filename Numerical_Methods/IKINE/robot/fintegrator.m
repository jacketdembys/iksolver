function f=fintegrator(t,i)
%i==0--> derivative
% else normal value
h=0.001;
it=int64((t-0.5)/h+1)
if it>1000
    hola=1;
end
[qd,dqd,ddqd] = jtraj(0.2, 0.5, 1001);
%f=abs(log t ), t>0.5
if i==0
    %f=1/t;
    f=ddqd(it,1);
else
    f=dqd(it,1);
end