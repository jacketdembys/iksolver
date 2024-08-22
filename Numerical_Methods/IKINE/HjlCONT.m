function H=HjlCONT(q,Qlim,beta)
beta=(Qlim(:,2)-Qlim(:,1))*beta;

for i=1:size(q,1)
    if or(q(i)>=Qlim(i,2),q(i)<=Qlim(i,1))
        hb(i)=1;
    elseif and(q(i)>Qlim(i,1),q(i)<Qlim(i,1)+beta(i))
        hb(i)=fubeta(beta(i)+Qlim(i,1)-q(i),beta(i));
    elseif and(q(i)<Qlim(i,2),q(i)>Qlim(i,2)-beta(i))
        hb(i)=fubeta(beta(i)-Qlim(i,2)+q(i),beta(i));    
    else
        hb(i)=0;
    end
    
    
end

for i=1:size(q,1)
   if abs(hb(i))<0.000001
       hb(i)=0;      
    
   end
end
H=diag(ones(1,size(q,1))-hb);
%H=diag(hb);

