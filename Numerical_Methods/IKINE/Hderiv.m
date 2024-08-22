function dH=Hderiv(q,Qlim)
%Qlim(4,1)=0;
%Qlim(6,1)=-1.5;
m=size(Qlim,1);
for i=1:m
   dH(i)=1/(m*2)*((Qlim(i,2)-Qlim(i,1))^2*(2*q(i)-Qlim(i,2)-Qlim(i,1))/((Qlim(i,2)-q(i))^2*(q(i)-Qlim(i,1))^2));
  % dH2(i)=0.5*(sign(q(i)-0.5*(Qlim(i,1)+Qlim(i,2)))+sign(dH(i)))*dH(i)+1;
end

