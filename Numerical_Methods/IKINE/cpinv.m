function J1=cpinv(J,H)

hi=diag(H);
J1=zeros(size(J))';
%Calcul Xp
m=size(J,1);
%m=size(J,2);
hbin=zeros(m,1);

for i1=0:2^m-1
    %cada i es un Xp
    hbin=zeros(m,1);
    V=i1;
    
    if V>63
        hbin(7)=1;
        V=V-64;
    else
    end
    
    if V>31
        hbin(6)=1;
        V=V-32;
    else
    end
    
    if V>15
        hbin(5)=1;
        V=V-16;
    else
    end
    
    if V>7
        hbin(4)=1;
        V=V-8;
    else
    end
    
    if V>3
        hbin(3)=1;
        V=V-4;
    else
    end
    
    if V>1
        hbin(2)=1;
        V=V-2;
    else
    end
    
    if V>0
        hbin(1)=1;
        V=V-1;
    else
    end
     
    H0=diag(hbin);
    
    produ=1;
    for i2=1:size(H,1)
        if hbin(i2)==1
            produ=produ*hi(i2);
        else
            produ=produ*(1-hi(i2));
        end
    end
 
if produ>0
    J1=J1+produ*pinv(H0*J);
end
   %J1=J1+produ*pinv(J*H0);
    
    
    h=1;
end
