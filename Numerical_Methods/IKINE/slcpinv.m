function [J1,Mb]=slcpinv(J,H)
%Hb=eye(size(H,1))-H;
hi=diag(H);
J1=zeros(size(J))';
%Calcul Xp
%m=size(J,1);
m=size(J,2);
hbin=zeros(m,1);
[uu,ss,vv]=svd(J);
Mb=zeros(size(J,1),1);
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
    for i2=1:size(H,2)
        if hbin(i2)==1
            produ=produ*hi(i2);
        else
            produ=produ*(1-hi(i2));
        end
    end
 
   %tenim el producte i H0, calculem el jacobia J1=J^[H+]
   J1=J1+produ*pinv(J*H0);
   
   %calculem el terme variable
   for a=1:size(J,1)
       VJ=0;
       for ia=1:m
            VJ=VJ+abs(vv(ia,a))*norm(J(:,ia));           
       end
        Mb(a)=Mb(a)+H0(a,a)/ss(a,a)*VJ;  
   end
   
    h=1;
end
