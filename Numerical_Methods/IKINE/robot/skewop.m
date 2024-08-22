function M=skewop(w)
M=zeros(3);
M(1,2)=-w(3);
M(2,1)=w(3);

M(1,3)=w(2);
M(3,1)=-w(2);

M(2,3)=-w(1);
M(3,2)=w(1);

