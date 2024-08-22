function A=mappingmatrix(x,y)
%returns a matrix verifying A*x=y
n=size(x,1);
nx=norm(x);
A=eye(n);
if norm(x)>0.000001
    for i=1:n
       notfound=1;
       j=1;
       while notfound==1
        k=A(i,:);
        %k=[ones(1,j) zeros(1,n-j)];%rand(1,n)-ones(1,n)*0.5;
        %k=k/norm(k);

        if abs(k*x)>0.00000001
           A(i,:)=k*y(i,1)/(k*x);
           notfound=0;
        else
           A(i,j)=1; 
        end
        j=j+1;
       end


    end
end