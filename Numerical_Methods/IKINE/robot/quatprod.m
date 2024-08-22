function out=quatprod(Q1,Q2)
%with notation [eta;eps]
eta1=Q1(1);
eta2=Q2(1);
eps1=Q1(2:4);
eps2=Q2(2:4);
out=[eta1*eta2-eps1'*eps2;eta1*eps2+eta2*eps1+cross(eps1,eps2);