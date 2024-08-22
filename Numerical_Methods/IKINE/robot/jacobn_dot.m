function dJ=jacobn_dot(robot,q,dq) 
dof=robot.n;
L=robot.link;
%FALTA DEFINIR W
dJ=zeros(6,dof);
R=zeros(dof,3,3);
p=zeros(3,dof);
dp=zeros(3,dof);
w=zeros(3,dof);
z=zeros(3,dof);
T=zeros(dof,4,4);

      
      z0=[0;0;1];
      Ti=L{1}(q(1));
      z(1:3,1)=Ti(1:3,3);
      T(1,:,:) = Ti;
      R(1,:,:)=Ti(1:3,1:3);
      p(:,1)=Ti(1:3,4);
      w(:,1)=dq(1)*z0;
      dp(:,1)=cross(w(:,1),z0);
        
   for i=2:dof
      %compute Ti and split into R,z and p
      Ti=L{i}(q(i));  
      T(i,:,:)=squeeze(T(i-1,:,:))*Ti; %T0i
      R(i,:,:) =squeeze(T(i,1:3,1:3));
      z(:,i)=squeeze(R(i,1:3,3))';
      p(:,i)=squeeze(T(i,1:3,4))';
       
      %compute wi, dpi
      w(:,i)=w(:,i-1)+dq(i)*z(:,i-1); %wi=wi-1+dqi*zi-1
      dp(:,i)=dp(:,i-1)+cross(w(:,i),(p(:,i)-p(:,i-1)));
      
           
   end
   
   
  dJ=[cross(z0,dp(:,dof));zeros(3,1)];
  
   for j=2:dof
        dzj1=skewop(w(:,j-1))*squeeze(R(j-1,:,:))*[0;0;1]; %dz(j-1)=S(wj-1)*Rj-1*[0;0;1]
        aux=cross(dzj1,p(:,dof)-p(:,j-1))+cross(z(:,j-1),dp(:,dof)-dp(:,j-1));
        dJ=[dJ [aux;dzj1]];
   end
   

%   if(ref != 0) {
%      Matrix zeros(3,3);
%      zeros = (Real) 0.0;
%      Matrix RT = R[ref].t();
%      Matrix Rot;
%      Rot = ((RT & zeros) | (zeros & RT));
%      jacdot = Rot*jacdot;
 %  }
%   jacdot.Release(); return jacdot;
%}