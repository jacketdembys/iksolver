function out=SelecDamp(J,in)
        nu=10;
        smin=0.01;
        m=7;
        n=7;
        
        v=null(J);
        [uu,S,vv]=svd(J);
        S2=zeros(6,7);
        for j=1:6
            s=S(j);
            s2=(((s+nu)*s+2)*s+2*smin)/((s+nu)*s+2);
            S2(j,j)=1/s2;
        end   
        Jf=uu*S2*vv';
        Jp=vv*pinv(S2)*uu';
        Ja=[Jp v];
        W=inv(Ja);
        Wi=[Jf;v'];
        Ki=diag([5 5 5 2 2 2 1]);
        e2=Ki*Wi*in;
        gammamax=[5 5 5 3 1 1 1];
        
        uu=[uu zeros(6,1);zeros(1 6) 1];
        
        w=zeros(m,1);
        for i=1:m
       
           if (i>n)
                wi=zeros(m,1);
                Ni=0;
                sig=0;
           else
               sig=1/S2(i,i);
                wi=sig*vv(:,i)*uu(:,i)'*e2;
                Ni=norm(uu(:,i));
           end
                    Mi=0;
           for ij=1:m
              Mi=Mi+sig*abs(vv(ij,i))*norm(Jf(:,i)); 
           end
           gammai=gammamax*min(1,Ni/Mi);
           
           if max(wi)>gammai
               thi=gammai*wi/max(wi); 
           else
               thi=wi;
           end         
           w=w+thi;
        end
        
        
           if max(w)>gammamax
               dq=gammamax*w/max(w); 
           else
               dq=w;
           end
           
           out=dq;