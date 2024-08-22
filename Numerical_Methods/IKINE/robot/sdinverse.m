function v=sdinverse(J,u,gammamax)
m=size(J,1);
%J square matrix mxm
[UU,SS,VV]=svd(J);
w=zeros(m,1);

for i=1:m
       
%    if (i>n)
%        wi=zeros(m,1);
%        Ni=0;
%        sig=0;
%    elseif and(i==2,SS(2,2)==0)
%        wi=zeros(m,1);
%        Ni=0;
%        sig=0;
%    else
%        sig=1/SS(i,i);
%        wi=sig*VV(:,i)*UU(:,i)'*u;
%        Ni=norm(UU(:,i));
%    end
           
    sig=1/SS(i,i);
    wi=sig*VV(:,i)*UU(:,i)'*u;
    Ni=norm(UU(:,i));
    Mi=0;
           
           
    for ij=1:m
        Mi=Mi+sig*abs(VV(ij,i))*norm(J(:,i)); 
    end
    
    gammai=gammamax*min(1,Ni/Mi);
           
    if max(abs(wi))>gammai
        thi=gammai*wi/max(abs(wi)); 
    else
        thi=wi;
    end         
    w=w+thi;
           
end

if max(abs(w))>gammamax
    v=gammamax*w/max(abs(w)); 
else
    v=w;
end
           
           
           