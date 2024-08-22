function [qfinal,efinal,iterations,time,maxdqnorm]=METHODSit(a,t,st,itmax,emax,qi,qo,To)
%close all
% a is the method used (integer, see code below)
% t is the experimental setup used (different cases for some methods)
% st was the step selection in some methods
% itmax the max number of iterations
% emax the maximum error admited for a solution
% qi, qo are joint positions provided to set the trajectory initial and
% final points, can be overwritten depending on t.
% To is the objective homogeneous transform
%% ROBOT DEFINITIONS


r=WAMarm7;
Qlim=zeros(7,2);
Qlim(1,1)=-2.6;
Qlim(1,2)=2.6;
Qlim(2,1)=-2.0;
Qlim(2,2)=2.0;
Qlim(3,1)=-2.8;
Qlim(3,2)=2.8;
Qlim(4,1)=-0.9;
Qlim(4,2)=3.1;
Qlim(5,1)=-4.8;
Qlim(5,2)=1.3;
Qlim(6,1)=-1.6;
Qlim(6,2)=1.6;
Qlim(7,1)=-2.2;
Qlim(7,2)=2.2;

maxdqnorm=0;

minimacount=0;

%% PARAMETERS



%% INITIAL CONDITIONS
%t=0;
if t==0 %singularity
    %situ='Vertical singularity end, ';
    qi=[0 0 0 2.2 0 0 0]';
    qo=[0 0 0 0 0 0 0]';
    To=fkine(r,qo);
    
    %aux=tr2eul(To);
    %xgd=[To(1:3,4);tr2eul(To)'];
    Td=To;
    %xgd=[0.0;0.1;0;0;0;0];
elseif t==1 %joint limits
    %situ='Joint Limit situation, ';
    qi=zeros(7,1);
    %xgd=[0.0;0.1;0;0;0;0];
    
elseif t==2 %singularity horizontal
    %situ='Horizontal singularity end, ';
    qi=zeros(7,1);
    %xgd=[0.0;0.1;0;0;0;0];
elseif t==3 %singularity horizontal
    %situ='Horizontal singularity end, ';
    %  T=fkine(r,qi);
    %aux=tr2eul(To);
    %xgd=[To(1:3,4);tr2eul(To)'];
    Td=To;
    %xgd=[0.0;0.1;0;0;0;0];
else %RANDOM
    %situ='Random end, '
    %qo=MostraRand(Qlim);
    %qi=MostraRand(Qlim);
    %qi=randprova(4,pi)
    To=fkine(r,qo);
    %    T=fkine(r,qi);
    %aux=tr2eul(To);
    %xgd=[To(1:3,4);tr2eul(To)'];
    Td=To;
end


%% METHOD SELECTION
%    if a==0;
%        meth='Jacobian Transpose, ' % JACOBIAN TRANSPOSE
%    elseif a==1
%        meth='Jacobian Pseudoinverse, ' % JACOBIAN PSEUDOINVERSE
%    elseif a==2
%        meth='Damped jacobian, ' %DAMPED JACOBIAN
%    elseif a==3
%        meth='Filtered jacobian, ' %DAMPED JACOBIAN
%    elseif a==4
%        meth='Error Damped jacobian, ' %WEIGHTED JACOBIAN
%    elseif a==5
%        meth='Improved error damped, ' %WEIGHTED JACOBIAN
%    elseif a==6
%        meth='Smooth Filtering, ' %WEIGHTED JACOBIAN
%    elseif a==7
%        meth='Smooth Filtering + ED, ' %WEIGHTED JACOBIAN
%    elseif a==8
%        meth='Selectively-damped'
%    elseif a==9
%        meth='Jacobian Weighting, ' %WEIGHTED JACOBIAN
%    elseif a==10
%        meth='Joint Clamping, ' %JOINT CLAMPING
%    elseif a==12
%        meth='Gradient Projection, ' %JOINT CLAMPING
%    elseif a==13
%        meth='Task Priority, ' %JOINT CLAMPING
%    elseif a==14
%        meth='Continuous Task Priority, ' %JOINT CLAMPING
%    elseif a==15
%        meth='Continuous Task Priority+SF, ' %JOINT CLAMPING
%    elseif a==16
%        meth='Continuous Task Priority+SD, ' %JOINT CLAMPING
%    elseif a==17
%        meth='Continuous Task Priority+SD+SF, ' %JOINT CLAMPING
%    end


%% STEP SELECTION%

%if st==0
%    sstep='0.1 step'; % alpha=0.1
%elseif st==1
%    sstep='least eig value step'; % alpha=smaller eigenvalue
%elseif st==2
%    sstep='step=1';% trying to make alpha·J·(J*·e) the same magnitude of e:
%end%alpha=dot(J·(J*·e),e)/dot(J·(J*·e),J·(J*·e))



%% INITIALIZATION OF VARIABLES
m=size(qi,1);
n=6; %task size - cartesian space dimension-

final=0;
q=qi;
% noise can be added:
%q=q+(rand(m,1)-0.5)*0.000001
%q=q-rand(m,1)*0.0001
QQ=q';
T=fkine(r,q);


% MODIFY ACCORDING TO TASK IF NECESSARY
%xg=[T(1:3,4);tr2eul(T)'];
%xgi=xg;
STDev=[];
xg=T(1:3,4);
%xgi=xg;
xgd=Td(1:3,4);
ex=xgd-xg;
eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
%e2=[ex;eo];
EE=[ex' eo' norm(ex)+norm(eo)];
%XX=[xg'];
it=0;
%AA=[];
t0=clock;
lambda=0.005;
lambdamax=0.02;
mu=0.2;
%eps=0.01;
nu=5;
gammamax=0.5;
hc=-15;
eps=5;
s=4;

%VARIABLE INDICATING if we plot things
plotegem=0;
if plotegem==1
    titol='plotting';
    AA=[];
    XX=xg';
    xgi=xg;
    ex=xgd-xg;
    eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
    EE=[ex' eo' norm(ex)+norm(eo)];
    QQ=q';
end
%figure(1)
%titol=[meth situ sstep]

dH0=Hderiv(q,Qlim)';

%% ITERATIONS
while final==0
    
    
    if a==0
        %JACOBIAN TRANSPOSE
        %method
        it=it+1;
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J2=jacob0(r,q);
        
        %step
        JJTe=J2*J2'*e2;
        alpha=dot(JJTe,e2)/dot(JJTe,JJTe);
        
        %actualization
        %q=pi2pi(q+alpha*J2'*e2);
        dq=alpha*J2'*e2;
        q=q+dq;
        %q = pi2piD(q);
        
    elseif a==1
        %JACOBIAN PSEUDOINVERSE
        %method
        it=it+1;
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J=jacob0(r,q);
        Ju=pinv(J);
        
        %         %step
        %         if st==0
        %             alpha=0.1;
        %         elseif st==1
        %             aux=svd(J);
        %             alpha=0.2*max(min(aux),0.05);
        %             %alpha=max(min(aux),0.02);
        %         elseif st==2
        %             %JJe=J*Ju*e2;
        %             %alpha=dot(JJe,e2)/dot(JJe,JJe);
        %             alpha=1;
        %         end
        alpha=1;
        
        %actualization
        %q=pi2piD(q+alpha*Ju*e2,Qlim);
        dq=alpha*Ju*e2;
        q=q+dq;
        
    elseif a==2
        %DAMPED JACOBIAN
        %method
        it=it+1;
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J=jacob0(r,q);
        [UU,SS,VV]=svd(J);
        
        %step
        if st==0
            alpha=0.1;
        elseif st==1
            aux=svd(J);
            alpha=0.2*max(min(aux),0.05);
            %alpha=max(min(aux),0.02);
        elseif st==2
            %JJe=J*Ju*e2;
            %alpha=dot(JJe,e2)/dot(JJe,JJe);
            alpha=1;
        end
        
        Ju=zeros(size(J))';
        for i=1:n
            SS(i,i)=SS(i,i)/(SS(i,i)^2+lambda^2);
            Ju=Ju+VV(:,i)*UU(:,i)'*SS(i,i);
        end
        
        dq=alpha*Ju*e2;
        
        q=q+dq;
        
    elseif a==3
        
        %FILTERED JACOBIAN
        %method
        it=it+1;
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J=jacob0(r,q);
        [UU,SS,VV]=svd(J);
        lambda2=lambdamax^2*(1-(SS(n,n)/eps)^2);
        %SS2=SS+lambda2^2*aux;
        Ju=zeros(size(J))';
        
        mu=0;
        %step
        if st==0
            alpha=0.1;
        elseif st==1
            aux=svd(J);
            alpha=0.2*max(min(aux),0.05);
            %alpha=max(min(aux),0.02);
        elseif st==2
            %JJe=J*Ju*e2;
            %alpha=dot(JJe,e2)/dot(JJe,JJe);
            alpha=1;
        end
        
        %dq=0;
        SS2=zeros(n);
        for i=1:n-1
            SS2(i,i)=1/SS(i,i);
        end
        SS2(n,n)=SS(n,n)/(SS(n,n)^2+lambda2^2);
        for i=1:n
            Ju=Ju+VV(:,i)*UU(:,i)'*SS2(i,i);
        end
        dq=alpha*Ju*e2;
        
        q=q+dq;
        
    elseif a==4 %ERROR DAMPED
        
        it=it+1;
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J=jacob0(r,q);
        [UU,SS,VV]=svd(J);
        
        %step
        if st==0
            alpha=0.1;
        elseif st==1
            aux=svd(J);
            alpha=0.2*max(min(aux),0.05);
            %alpha=max(min(aux),0.02);
        elseif st==2
            %JJe=J*Ju*e2;
            %alpha=dot(JJe,e2)/dot(JJe,JJe);
            alpha=1;
        end
        
        Ju=zeros(size(J))';
        
        for i=1:n
            SS(i,i)=SS(i,i)/(SS(i,i)^2+0.5*(e2'*e2));
            Ju=Ju+VV(:,i)*UU(:,i)'*SS(i,i);
        end
        
        dq=alpha*Ju*e2;
        
        q=q+dq;
        
    elseif a==5 %IMPROVED ERROR DAMPED
        it=it+1;
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J=jacob0(r,q);
        [UU,SS,VV]=svd(J);
        
        %step
        if st==0
            alpha=0.1;
        elseif st==1
            aux=svd(J);
            alpha=0.2*max(min(aux),0.05);
            %alpha=max(min(aux),0.02);
        elseif st==2
            %JJe=J*Ju*e2;
            %alpha=dot(JJe,e2)/dot(JJe,JJe);
            alpha=1;
        end
        
        Ju=zeros(size(J))';
        
        for i=1:n
            SS(i,i)=SS(i,i)/(SS(i,i)^2+0.5*(e2'*e2)+0.001);
            Ju=Ju+VV(:,i)*UU(:,i)'*SS(i,i);
        end
        
        dq=alpha*Ju*e2;
        
        q=q+dq;
        
    elseif a==6 %SMOOTH FILTERING
        
        %method
        
        smin=0.01;
        it=it+1;
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J=jacob0(r,q);
        
        
        [UU,SS,VV]=svd(J);
        Ju=zeros(size(J))';
        SS2=zeros(size(SS));
        for i=1:n
            SS2(i,i)=eigfilter(nu,smin,SS(i,i));
            Ju=Ju+VV(:,i)*UU(:,i)'/SS2(i,i);
        end
        %Jb=UU*SS2*VV';
        
        mu=0;
        %step
        if st==0
            alpha=0.1;
        elseif st==1
            aux=svd(J);
            alpha=0.2*max(min(aux),0.05);
            %alpha=max(min(aux),0.02);
        elseif st==2
            %JJe=J*Ju*e2;
            %alpha=dot(JJe,e2)/dot(JJe,JJe);
            alpha=1;
        end
        
        
        dq=alpha*Ju*e2;
        
        q=q+dq;
        
    elseif a==7
        %SMOOTH FILTERING+ED
        
        %method
        
        smin=0.01;
        it=it+1;
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J=jacob0(r,q);
        
        
        [UU,SS,VV]=svd(J);
        Ju=zeros(size(J))';
        SS2=zeros(size(SS));
        for i=1:n
            SS2(i,i)=eigfilter(nu,smin,SS(i,i));
            Ju=Ju+VV(:,i)*UU(:,i)'*SS2(i,i)/(SS2(i,i)^2+0.5*(e2'*e2));
        end
        %Jb=UU*SS2*VV';
        
        mu=0;
        %step
        if st==0
            alpha=0.1;
        elseif st==1
            aux=svd(J);
            alpha=0.2*max(min(aux),0.05);
            %alpha=max(min(aux),0.02);
        elseif st==2
            %JJe=J*Ju*e2;
            %alpha=dot(JJe,e2)/dot(JJe,JJe);
            alpha=1;
        end
        
        
        dq=alpha*Ju*e2;
        %q=pi2piD(q+dq,Qlim);
        q=q+dq;
        
    elseif a==8
        %SELECTIVELY DAMPED
        %method
        
        %smin=0.01;
        it=it+1;
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J=jacob0(r,q);
        [UU,SS,VV]=svd(J);
        
        w=zeros(m,1);
        for i=1:m
            
            if (i>n)
                wi=zeros(m,1);
                Ni=0;
                sig=0;
            elseif and(i==2,SS(2,2)==0)
                wi=zeros(m,1);
                Ni=0;
                sig=0;
            else
                sig=1/SS(i,i);
                wi=sig*VV(:,i)*UU(:,i)'*e2;
                Ni=norm(UU(:,i));
            end
            Mi=0;
            for ij=1:m
                Mi=Mi+sig*abs(VV(ij,i))*norm(J(:,i));
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
        mu=0;
        %step
        if st==0
            alpha=0.1;
        elseif st==1
            aux=svd(J);
            alpha=0.2*max(min(aux),0.05);
            %alpha=max(min(aux),0.02);
        elseif st==2
            %JJe=J*Ju*e2;
            %alpha=dot(JJe,e2)/dot(JJe,JJe);
            alpha=1;
        end
        %actualization
        %dH=Hderiv(q,Qlim);
        %        dq=alpha*(Ju*e2+eye(4)*Ju*J)*dH;
        q=q+dq;
        
    elseif a==9
        %JACOBIAN WEIGHTING
        %method
        
        it=it+1;
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J=jacob0(r,q);
        
        dH1=Hderiv(q,Qlim)';
        DH=abs(dH1)-abs(dH0);
        
        %step
        if st==0
            alpha=0.1;
        elseif st==1
            aux=svd(J);
            alpha=0.2*max(min(aux),0.05);
            %alpha=max(min(aux),0.02);
        elseif st==2
            %JJe=J*Ju*e2;
            %alpha=dot(JJe,e2)/dot(JJe,JJe);
            alpha=1;
        end
        
        for i=1:m
            if DH(i)>0
                w(i)=1+abs(dH1(i));
            else
                w(i)=1;
            end
        end
        
        Wisqrt=diag(1./sqrt(w));
        Ju=pinv(J*Wisqrt);
        
        dq=Wisqrt*Ju*e2;
        
        %actualization
        
        q=q+dq;
        
        dH0=dH1;
        
    elseif a==10
        %JOINT CLAMPING
        %method
        [H,q]=clampjoints(q,Qlim);
        it=it+1;
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J=jacob0(r,q);
        %dH1=Hderiv(q,Qlim)';
        %DH=abs(dH1)-abs(dH0);
        %step
        if st==0
            alpha=0.1;
        elseif st==1
            aux=svd(J);
            alpha=0.2*max(min(aux),0.05);
            %alpha=max(min(aux),0.02);
        elseif st==2
            %JJe=J*Ju*e2;
            %alpha=dot(JJe,e2)/dot(JJe,JJe);
            alpha=1;
        end
        Ju=pinv(J*H);
        dq=H*Ju*e2;
        %actualization
        q=q+dq;
        
    elseif a==11
        %JOINT CLAMPING with continuous activation
        %method
        beeta=0.1;
        H=HjlCONT(q,Qlim,beeta);
        it=it+1;
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J=jacob0(r,q);
        %dH1=Hderiv(q,Qlim)';
        %DH=abs(dH1)-abs(dH0);
        %step
        if st==0
            alpha=0.1;
        elseif st==1
            aux=svd(J);
            alpha=0.2*max(min(aux),0.05);
            %alpha=max(min(aux),0.02);
        elseif st==2
            %JJe=J*Ju*e2;
            %alpha=dot(JJe,e2)/dot(JJe,JJe);
            alpha=1;
        end
        Ju=pinv(J*H);
        dq=H*Ju*e2;
        %actualization
        q=q+dq;
        %q=q+dq;
        
    elseif a==12
        %GRADIENT PROJECTION
        %method
        it=it+1;
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J=jacob0(r,q);
        Ju=pinv(J);
        
        %step
        if st==0
            alpha=0.1;
        elseif st==1
            aux=svd(J);
            alpha=0.2*max(min(aux),0.05);
            %alpha=max(min(aux),0.02);
        elseif st==2
            %JJe=J*Ju*e2;
            %alpha=dot(JJe,e2)/dot(JJe,JJe);
            alpha=1;
        end
        
        %actualization
        dH=Hderiv(q,Qlim);
        %[mgrad,manip]=gradMANIP2R(q,0.0001,r)
        dq=alpha*Ju*e2+mu*alpha*(eye(m)-Ju*J)*dH';
        %dq=alpha*Ju*e2+mu*alpha*(eye(4)-Ju*J)*mgrad;
        q=q+dq;
        
        
    elseif a==13
        %TASK PRIORITY CLAMPING
        %method
        lambdajl=0.2;
        beeta=0.1;
        it=it+1;
        J1=eye(7)-HjlCONT(q,Qlim,beeta);
        e1=-lambdajl*q;
        q1=pinv(J1)*e1;
        P1=eye(7)-pinv(J1)*J1;
        
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J2=jacob0(r,q);
        
        
        q2=q1+pinv(J2*P1)*(e2);%+J2*q1);
        %step
        
        if st==0
            alpha=0.1;
        elseif st==1
            aux=svd(J2);
            alpha=0.2*max(min(aux),0.05);
            %alpha=max(min(aux),0.02);
        elseif st==2
            %JJe=J*Ju*e2;
            %alpha=dot(JJe,e2)/dot(JJe,JJe);
            alpha=1;
        end
        dq=alpha*q2;
        %actualization
        %dH=Hderiv(q,Qlim);
        %       dq=alpha*(Ju*e2+eye(4)*Ju*J)*dH;
        q=q+dq;
        
    elseif a==14
        %CONTINUOUS TASK PRIORITY CLAMPING +CONTINUOUS ACTIVATION
        %method
        lambdajl=0.2;
        %smin=0.01;
        beeta=0.2;
        it=it+1;
        J1=eye(7);
        H=eye(7)-HjlCONT(q,Qlim,beeta);
        e1=-lambdajl*q;
        q1=rcpinv(J1,H)*e1;
        P1=eye(7)-H;%rcpinv(J1,H)*J1;
        %P1=eye(4)-pinv(J1)*J1;
        
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J2=jacob0(r,q);
        
        [UU,SS,VV]=svd(J2);
        %Ju=zeros(size(J2))';
        
        J2=UU*SS*VV';
        J2P1=lcpinv(J2,P1);
        %step
        if st==0
            alpha=0.1;
        elseif st==1
            aux=svd(J2);
            alpha=0.2*max(min(aux),0.05);
            %alpha=max(min(aux),0.02);
        elseif st==2
            %JJe=J*Ju*e2;
            %alpha=dot(JJe,e2)/dot(JJe,JJe);
            alpha=1;
        end
        q2=q1+J2P1*(alpha*e2-J2*q1);
        dq=alpha*q2;
        %actualization
        %dH=Hderiv(q,Qlim);
        %       dq=alpha*(Ju*e2+eye(4)*Ju*J)*dH;
        q=q+dq;
        jlrespect=1;
        for j=1:7
            if or(q(j)>Qlim(j,2),q(j)<Qlim(j,1))
                jlrespect=0;
                break
            end
        end
        
        
        
    elseif a==15
        %CONTINUOUS TASK PRIORITY CLAMPING +CONTINUOUS ACTIVATION+SF
        %method
        lambdajl=0.2;
        smin=0.01;
        beeta=0.2;
        it=it+1;
        J1=eye(7);
        H=eye(7)-HjlCONT(q,Qlim,beeta);
        e1=-lambdajl*q;
        q1=rcpinv(J1,H)*e1;
        P1=eye(7)-H;%rcpinv(J1,H)*J1;
        %P1=eye(4)-pinv(J1)*J1;
        
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J2=jacob0(r,q);
        
        [UU,SS,VV]=svd(J2);
        %Ju=zeros(size(J2))';
        SS2=zeros(size(SS));
        for i=1:n
            SS2(i,i)=eigfilter(nu,smin,SS(i,i));
        end
        J2=UU*SS2*VV';
        J2P1=lcpinv(J2,P1);
        %step
        if st==0
            alpha=0.1;
        elseif st==1
            aux=svd(J2);
            alpha=0.2*max(min(aux),0.05);
            %alpha=max(min(aux),0.02);
        elseif st==2
            %JJe=J*Ju*e2;
            %alpha=dot(JJe,e2)/dot(JJe,JJe);
            alpha=1;
        end
        q2=q1+J2P1*(alpha*e2-J2*q1);
        dq=alpha*q2;
        %actualization
        %dH=Hderiv(q,Qlim);
        %       dq=alpha*(Ju*e2+eye(4)*Ju*J)*dH;
        q=q+dq;
        jlrespect=1;
        for j=1:7
            if or(q(j)>Qlim(j,2),q(j)<Qlim(j,1))
                jlrespect=0;
                break
            end
        end
        
    elseif a==16
        %CONTINUOUS TASK PRIORITY CLAMPING: TP+CA+SD
        %method
        
        
        lambdajl=0.2;
        %smin=0.01;
        beta=0.05;
        it=it+1;
        J1=eye(7);
        H=eye(7)-HjlCONT(q,Qlim,beta);
        e1=-lambdajl*q;
        q1=rcpinv(J1,H)*e1;
        P1=eye(7)-H;%rcpinv(J1,H)*J1;
        %P1=eye(4)-pinv(J1)*J1;
        
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        
        
        
        J=jacob0(r,q);
        [UU,SS,VV]=svd(J);
        %Ju=zeros(size(J))';
        
        J2=UU*SS*VV';
        
        %[J2P1,Mb2]=slcpinv(J2,P1);
        J2P1=slcpinv(J2,P1);
        [vv3,ss3,uu3]=svd(J2P1);
        Mb=zeros(7,1);
        for ip1=1:n
            for ip2=1:m
                Mb(ip1,1)=Mb(ip1,1)+ss3(ip1,ip1)*abs(vv3(ip2,ip1))*norm(J2(:,ip2));
            end
        end
        %selectively damped least squares
        for i=1:n
            wi(:,i)=ss3(i,i)*vv3(:,i)*uu3(:,i)'*e2;
        end
        %J2P1=lcpinv(J2,P1);
        w=zeros(m,1);
        for ig=1:n
            gammai=gammamax*min(1,1/Mb(ig));
            
            if max(wi(:,ig))>gammai
                thi=gammai*wi(:,ig)/max(wi(:,ig));
            else
                thi=wi(:,ig);
            end
            w=w+thi;
            
        end
        
        if max(w)>gammamax
            dq=gammamax*w/max(w);
        else
            dq=w;
        end
        
        q2=q1-J2P1*J2*q1;
        dq=dq+q2;
        
        if norm(dq)>gammamax
            dq=dq/norm(dq)*gammamax;
        end
        alpha=1;
        %dqnorm=norm(dq);
        %if dqnorm>gammalim
        %   dq=dq/dqnorm*gammalim;
        %end
        %actualization
        %dH=Hderiv(q,Qlim);
        %       dq=alpha*(Ju*e2+eye(4)*Ju*J)*dH;
        q=q+dq;
        
        jlrespect=1;
        for j=1:7
            if or(q(j)>Qlim(j,2),q(j)<Qlim(j,1))
                jlrespect=0;
                break
            end
        end
        
        
    elseif a==17
        %CONTINUOUS TASK PRIORITY CLAMPING: TP+CA+SF+SD
        %method
        
        
        lambdajl=0.2;
        smin=0.01;
        beta=0.05;
        it=it+1;
        J1=eye(7);
        H=eye(7)-HjlCONT(q,Qlim,beta);
        e1=-lambdajl*q;
        q1=rcpinv(J1,H)*e1;
        P1=eye(7)-H;%rcpinv(J1,H)*J1;
        %P1=eye(4)-pinv(J1)*J1;
        
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        
        J=jacob0(r,q);
        [UU,SS,VV]=svd(J);
        %Ju=zeros(size(J))';
        SS2=zeros(size(SS));
        for i=1:n
            %smoothly filtered eigenvalues
            SS2(i,i)=eigfilter(nu,smin,SS(i,i));
            
        end
        J2=UU*SS2*VV';
        
        %[J2P1,Mb2]=slcpinv(J2,P1);
        J2P1=slcpinv(J2,P1);
        [vv3,ss3,uu3]=svd(J2P1);
        Mb=zeros(7,1);
        for ip1=1:n
            for ip2=1:m
                Mb(ip1,1)=Mb(ip1,1)+ss3(ip1,ip1)*abs(vv3(ip2,ip1))*norm(J2(:,ip2));
            end
        end
        
        %selectively damped least squares
        for i=1:n
            wi(:,i)=ss3(i,i)*vv3(:,i)*uu3(:,i)'*e2;
        end
        %J2P1=lcpinv(J2,P1);
        w=zeros(m,1);
        for ig=1:n
            gammai=gammamax*min(1,1/Mb(ig));
            
            if max(wi(:,ig))>gammai
                thi=gammai*wi(:,ig)/max(wi(:,ig));
            else
                thi=wi(:,ig);
            end
            w=w+thi;
            
        end
        
        if max(w)>gammamax
            dq=gammamax*w/max(w);
        else
            dq=w;
        end
        
        q2=q1-J2P1*J2*q1;
        dq=dq+q2;
        
        if norm(dq)>gammamax
            dq=dq/norm(dq)*gammamax;
        end
        alpha=1;
        %dqnorm=norm(dq);
        %if dqnorm>gammalim
        %   dq=dq/dqnorm*gammalim;
        %end
        %actualization
        %dH=Hderiv(q,Qlim);
        %       dq=alpha*(Ju*e2+eye(4)*Ju*J)*dH;
        q=q+dq;
        
        jlrespect=1;
        for j=1:7
            if or(q(j)>Qlim(j,2),q(j)<Qlim(j,1))
                jlrespect=0;
                break
            end
        end
        
        
        
        
    elseif  a==18
        %GWLS
        %method
        it=it+1;
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J2=jacob0(r,q);
        
        alpha=1;
        
        Hg=Hfunct(q,Qlim);
        dHg=Hderiv(q,Qlim);
        dHg=dHg/norm(dHg);
        Kg=COMPorto(dHg);
        T=[dHg;Kg];
        
        Ti=pinv(T);
        k=0.2;
        Jv=J2*Ti;
        dqv=Jv'*pinv(Jv*Jv')*(k*e2);
        Hgpre=Hg+dqv(1)*1;
        de=hc-eps-Hgpre;
        
        if de>0
            wv1=(2*eps)^s/((2*eps)^s+de^s);
        else
            wv1=1;
        end
        
        %recalculate qv:
        %Wvi=diag([wv1 1 1 1 1 1 1]);
        Wvi=diag([wv1 1 1 1 1 1 1]);
        dqv=Wvi*Jv'*pinv(Jv*Wvi*Jv'+lambda^2*eye(6))*(k*e2);
        dq=Ti*dqv;
        %actualization
        %q=pi2pi(q+alpha*J2'*e2);
        q=q+dq;
        %q = pi2piD(q);
        
        
        
    elseif a==88
        %SELECTIVELY DAMPED+Eigenvalue filtering
        %method
        it=it+1;
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J2=jacob0(r,q);
        %smin=0.01;
        [UU,SS,VV]=svd(J2);
        for jj=1:size(SS,1)
            SS(jj,jj)=eigfilter(10,0.01,SS(jj,jj));
        end
        
        J2v=UU*SS*VV';
        w=zeros(m,1);
        for i=1:m
            
            if (i>n)
                wi=zeros(m,1);
                Ni=0;
                sig=0;
            elseif and(i==2,SS(2,2)==0)
                wi=zeros(m,1);
                Ni=0;
                sig=0;
            else
                sig=1/SS(i,i);
                wi=sig*VV(:,i)*UU(:,i)'*e2;
                Ni=norm(UU(:,i));
            end
            Mi=0;
            for ij=1:m
                Mi=Mi+sig*abs(VV(ij,i))*norm(J2v(:,i));
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
        mu=0;
        %step
        
        alpha=1;
        % AA=[AA;cond2(Ju)];
        %actualization
        %dH=Hderiv(q,Qlim);
        %        dq=alpha*(Ju*e2+eye(4)*Ju*J)*dH;
        q=q+dq;
        
        
        
        
        
        
        
        
        
    elseif a==-100
        
        %CONTINUOUS TASK PRIORITY CLAMPING with TASK clamp and smooth activation
        %method
        lambdajl=0.2;
        smin=0.01;
        beeta=0.2;
        it=it+1;
        J1=eye(7);
        H=eye(7)-HjlCONT(q,Qlim,beeta);
        e1=-lambdajl*q;
        q1=rcpinv(J1,H)*e1;
        P1=eye(7)-H;%rcpinv(J1,H)*J1;
        %P1=eye(4)-pinv(J1)*J1;
        
        
        
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J2=jacob0(r,q);
        
        G=eye(6);
        if norm(e2)>0.2
            G(4:6,4:6)=eye(3)*min(1,max(1-(0.6-norm(e2))/0.3,0));
        end
        
        [UU,SS,VV]=svd(J2);
        %Ju=zeros(size(J2))';
        SS2=zeros(size(SS));
        for i=1:n
            SS2(i,i)=eigfilter(nu,smin,SS(i,i));
        end
        J2=UU*SS2*VV';
        
        
        J2P1=LRpinv(J2,P1,G);
        
        %step
        
        if st==0
            alpha=0.1;
        elseif st==1
            aux=svd(J2);
            alpha=0.2*max(min(aux),0.05);
            %alpha=max(min(aux),0.02);
        elseif st==2
            %JJe=J*Ju*e2;
            %alpha=dot(JJe,e2)/dot(JJe,JJe);
            alpha=1;
        end
        
        q2=q1+J2P1*(alpha*e2-J2*q1);
        
        P2=P1-J2P1*J2;
        
        
        [grad,m3,dm]=gradMANIP(q,0.002,r);
        J3=[grad';grad';grad';grad';grad';grad';grad'];
        
        G3=[eye(6)-G zeros(6,1)];
        [Ug3,Sg3,Vg3]=svd(G3);
        %we sort the tasks on J3 on the unactivated tasks, so J2P1*J3b=0
        J3b=Ug3*Sg3*J3*Vg3';
        
        J3P2=lcpinv(J3b,P2);
        e3=0.1*max(0,0.02-m3)*ones(6,1);
        q3=q2+J3P2*(e3-J3b*q2);
        
        dq=alpha*q3;
        %actualization
        %dH=Hderiv(q,Qlim);
        %       dq=alpha*(Ju*e2+eye(4)*Ju*J)*dH;
        q=q+dq;
        jlrespect=1;
        for j=1:7
            if or(q(j)>Qlim(j,2),q(j)<Qlim(j,1))
                jlrespect=0;
                break
            end
        end
        
        
        
        
        
        
        %--------------------------------------------------------------------
        %----------------------------------------------------------
    end
    
    if plotegem==1
        XX=[XX;xg'];
        QQ=[QQ;q'];
        EE=[EE;ex' eo' norm([ex;eo])];
        AA=[AA;alpha];
    end
    T=fkine(r,q);
    %aux=tr2eul(T);
    %xg=[T(1:2,4);pi2pi(aux(3))];
    xg=T(1:3,4);
    
    ex=xgd-xg;
    eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
    e=[ex;eo];
    
    if or(norm(dq)<0.0001,or(norm(e)<emax,it>itmax))
        final=1;
    end
    %    plot(xgd(1),xgd(2),'*r');
    %    hold on
    %    plotroboprova(q,(it-1)/itmax)
    maxdqnorm=max(maxdqnorm,norm(dq));
    %XX=[XX;xg'];
    QQ=[QQ;q'];
    EE=[EE;ex' eo' norm(e)];
    localminimaescape=0;
    if(localminimaescape==1)
        
        STDev=[STDev;std(EE(max(1,it-20):it,7))];
        if and(STDev(it)<0.05,it>10)
            q=q+0.1*(minimacount+1)*(rand(7,1)-ones(7,1));
            minimacount=minimacount+1;
        end
    end
    %AA=[AA;alpha];
end


%  SHOW
qfinal=q;
efinal=e;
iterations=it;
time=-etime(t0,clock);



%% PLOT
if plotegem==1
    
    
    plot(QQ)
    hold on
    plot(STDev,'LineWidth',2)
    grid on
    legend('q1','q2','q3','q4','q5','q6','q7')
    Qlimlow=[Qlim(:,1)';Qlim(:,1)'];
    Qlimhig=[Qlim(:,2)';Qlim(:,2)'];
    Tlim=[1;it]
    plot(Tlim,Qlimlow,'Linewidth',0.5);
    hold on
    plot(Tlim,Qlimhig,'Linewidth',0.5);
    
    %TRAJECTORY
    figure(1)
    subplot(3,2,1)
    title 'Position trajectory'
    plot3(xgd(1),xgd(2),xgd(3),'*r');
    hold on
    plot3(xgi(1),xgi(2),xgi(3),'*m');
    for it2=1:it+1
        %       [P0,P1,P2,P3,P4,P5]=plotWAM(q,qlor,L)
        plotWAM(QQ(it2,:)',[1-(it2-1)/it,1-(it2-1)/it,(it2-1)/it],1);
    end
    
    xlabel 'x'
    ylabel 'y'
    zlabel 'z'
    title 'Trajectory'
    
    %JOINT TRAJECTORY
    ubound=0;
    lbound=0;
    for jj=1:size(QQ,2)
        for ii=1:size(QQ,1)
            ubound(ii,jj)=Qlim(jj,2);
            lbound(ii,jj)=Qlim(jj,1);
        end
    end
    
    
    %figure(2)
    subplot(3,2,2)
    title 'Joint trajectory joints 1,2'
    plot(QQ(:,1),'b','Linewidth',2);
    hold on
    plot(QQ(:,2),'g','Linewidth',2);
    xlabel 'iteration'
    ylabel 'qi (rad)'
    jj=1;
    qlor=rgb('DarkBlue');
    plot(ubound(:,jj),'Color',qlor,'Linewidth',1);
    plot(lbound(:,jj),'Color',qlor,'Linewidth',1);
    jj=2;
    qlor=rgb('DarkGreen');
    plot(ubound(:,jj),'Color',qlor,'Linewidth',1);
    plot(lbound(:,jj),'Color',qlor,'Linewidth',1);
    legend('q1','q2','max_1','min_1','max_2','min_2','Location','EastOutside')

    
    subplot(3,2,4)
    title 'Joint trajectory joints 3,4'
    plot(QQ(:,3),'r','Linewidth',2);
    hold on
    plot(QQ(:,4),'c','Linewidth',2);
    xlabel 'iteration'
    ylabel 'qi (rad)'
    jj=3;
    qlor=rgb('DarkRed');
    plot(ubound(:,jj),'Color',qlor,'Linewidth',1);
    plot(lbound(:,jj),'Color',qlor,'Linewidth',1);
    jj=4;
    qlor=rgb('DarkCyan');
    plot(ubound(:,jj),'Color',qlor,'Linewidth',1);
    plot(lbound(:,jj),'Color',qlor,'Linewidth',1);
    legend('q3','q4','max_3','min_2','max_4','min_4','Location','EastOutside')

    
    subplot(3,2,6)
    title 'Joint trajectory joints 5,6,7'
    plot(QQ(:,5),'Color',rgb('Purple'),'Linewidth',2);
    hold on
    plot(QQ(:,6),'Color',rgb('Khaki'),'Linewidth',2);
    plot(QQ(:,7),'k','Linewidth',2);
    xlabel 'iteration'
    ylabel 'qi (rad)'
    jj=5;
    qlor=rgb('Indigo');
    plot(ubound(:,jj),'Color',qlor,'Linewidth',1);
    plot(lbound(:,jj),'Color',qlor,'Linewidth',1);
    jj=6;
    qlor=rgb('DarkOliveGreen');
    plot(ubound(:,jj),'Color',qlor,'Linewidth',1);
    plot(lbound(:,jj),'Color',qlor,'Linewidth',1);
    jj=7;
    qlor=rgb('Gray');
    plot(ubound(:,jj),'Color',qlor,'Linewidth',1);
    plot(lbound(:,jj),'Color',qlor,'Linewidth',1);
    legend('q5','q6','q7','max_5','min_5','max_6','min_6','max_7','min_7','Location','EastOutside')%,'Location','NorthEastOutside')
   

    %ERROR
    subplot(3,2,3)
    % figure(3)
    plot(EE(:,7),'Linewidth',2)
    legend( 'error norm')
    title 'Absolute Error'
    
    %STEP
    %figure(4)
    subplot(3,2,5)
    plot(AA,'Linewidth',2)
    title 'Step'
    
    title_handle = subplot_title(titol);
    set(gcf, 'Position', get(0,'Screensize'));
end