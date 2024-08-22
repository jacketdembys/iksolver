function v=splitsignal(u,i,j)
n=size(u);
v=u(n/i*(j-1)+1:n/i*j);