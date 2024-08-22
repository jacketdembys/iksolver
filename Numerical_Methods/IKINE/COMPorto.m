function  Kg=COMPorto(dHg)
%dHg is a matrix with gradients as rows
%we complete with trivial unitary vectors, and we perform gramm schmidt
%orthonormalization of the part corresponding to Kg
m=size(dHg,2);
k=size(dHg,1);
E=eye(m);
Kg=[];
i=1;
while and(size(Kg,1)+k<m,i<8)
    
    ei=E(i,:);
    vi=ei;
    for j1=1:k
        vi=vi-dot(ei,dHg(j1,:))/dot(dHg(j1,:),dHg(j1,:))*dHg(j1,:);
    end
      
    for j2=1:size(Kg,1)
        vi=vi-dot(ei,Kg(j2,:))/dot(Kg(j2,:),Kg(j2,:))*Kg(j2,:);
    end
    
    if and(norm(vi)>0,(rank([dHg;Kg])<rank([dHg;Kg;vi])))
        vi=vi/norm(vi);
        Kg=[Kg;vi];
    else
    end
    
    i=i+1;  
    
    
end
