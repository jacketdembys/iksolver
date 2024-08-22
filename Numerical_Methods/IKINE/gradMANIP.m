function [grad,m,dm]=gradMANIP(q,h,wam7)
%q es el punt on es calcula el gradient
%robot es el robot
% dim es el nombre d'articulacions a tractar:4

grad=zeros(4,1);

J=jacob0(wam7,q);
m=sqrt(abs(det(J*J')));
dim=size(q,1);
E=eye(dim);


for i=1:7
    
    J=jacob0(wam7,q+E(i,:)'*h);
    %returns a 3x4 matrix
    man=sqrt(abs(det(J*J')));
    grad(i,1)=(man-m)/h;
end
dm=norm(grad);
grad=grad/dm;