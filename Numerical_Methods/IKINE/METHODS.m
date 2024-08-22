%function m4Rmethods(a,t,alpha)
close all

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







%% PARAMETERS
itmax=500;
emax=0.02;


%% INITIAL CONDITIONS
t=4;
if t==0 %singularity
    situ='Vertical singularity end, ';
    qi=[0 0 0 2.2 0 0 0]'+(rand(7,1)-0.5*ones(7,1))*0.01;
    qo=[0 0 0 0 0 0 0]';
    To=fkine(r,qo);
    
    aux=tr2eul(To);
    %xgd=[To(1:3,4);tr2eul(To)'];
    Td=To;
    xgd=[0.0;0.1;0;0;0;0];
elseif t==1 %joint limits
    situ='Joint Limit situation, ';
    qi=Qlim(1:7,1);
    xgd=[0.0;0.1;0;0;0;0];
    
elseif t==2 %singularity horizontal
    situ='Horizontal singularity end, ';
    qi=zeros(7,1);
    xgd=[0.0;0.1;0;0;0;0];
elseif t==3 % custom position
    %qi=zeros(7,1);
    qi=[0.8147;
        0.9058;
        0.1270;
        0.9134;
        0.6324;
        0.0975;
        0.2785];
    
    qo =[1.0882;
        1.3845;
        0.6094;
        0.9922;
        1.1177;
        0.5761;
        0.5212];
    To=fkine(r,qo);
    situ='inherited, ';
    %aux=tr2eul(To);
    %xgd=[To(1:3,4);tr2eul(To)'];
    Td=To;
    
else %RANDOM
    rng(1)
    situ='Random end, '
    %qo=MostraRand(Qlim);
    %qi=MostraRand(Qlim);

    qi = [-0.8031; -0.4129; 0.2174; 0.7768; -0.6202; -0.9458; 1.6637];
    qo = [-0.4315; 0.8813; -2.7994; 0.3093; -3.9048; -1.3045; -1.3805];
    %qi=randprova(4,pi)
    %To=fkine(r,qo);
    
    
    To=fkine(r,qo);
    T=fkine(r,qi);
    
    Td=To;
    %aux=tr2eul(To);
    %xgd=[To(1:3,4);tr2eul(To)'];
    %Td=To;
end


%% METHOD SELECTION
% a=0; meth='Jacobian Transpose, ' % JACOBIAN TRANSPOSE
 a=1; meth='Jacobian Pseudoinverse, ' % JACOBIAN PSEUDOINVERSE
% a=2; meth='Pinv gradient projection, ' % PINV GRADIENT PROJECTION
% a=3; meth='Damped jacobian, ' %DAMPED JACOBIAN
% a=4; meth='Filtered jacobian, ' %DAMPED JACOBIAN
% a=5; meth='Jacobian Weighting, ' %WEIGHTED JACOBIAN
% a=6; meth='Joint Clamping, ' %JOINT CLAMPING
% a=7; meth='Joint Clamping with cont act, ' %JOINT CLAMPING CONTINUOUS ACT.
% a=8; meth='Smoothly-filtered Jacobian, ' %JACOBIAN CONT FILT%
% a=9; meth='Selectively-damped'
% a=10; meth='Task priority clamping, ' %TASK PRIORITY CLAMPING
% a=11; meth='Task priority clamping with cont activation, ' %WEIGHTED JACOBIAN
% a=12; meth='Task priority clamping with cont activation and smooth filt., '
% a=6; meth='Extended Jacobian, ' %WEIGHTED JACOBIAN
% a=15;  meth='smooth, '
% a=8; meth='Continuous Clamping, ' %WEIGHTED JACOBIAN

%% STEP SELECTION%
%st=0; sstep='0.1 step' % alpha=0.1
%st=1; sstep='least eig value step' % alpha=menor vap
st=2; sstep='step=1'% trying to make alpha·J·(J*·e) the same magnitude of e:
%alpha=dot(J·(J*·e),e)/dot(J·(J*·e),J·(J*·e))



%% INITIALIZATION OF VARIABLES
m=size(qi,1);
n=6; %task size

final=0;
q=qi;
%q=q+(rand(m,1)-0.5)*0.000001
%q=q-rand(m,1)*0.0001
QQ=q';
T=fkine(r,q);
aux=tr2eul(T);

% MODIFY ACCORDING TO TASK
%xg=[T(1:3,4);tr2eul(T)'];
%xgi=xg;

xg=T(1:3,4);
xgi=xg;
xgd=Td(1:3,4);
ex=xgd-xg;
eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
e2=[ex;eo];
EE=[ex' eo' norm(ex)+norm(eo)];
XX=[xg'];
it=0;
AA=[];
t0=clock;
lambda=0.005;
lambdamax=0.02
mu=0.2;
eps=0.01
nu=5;
gammamax=0.5;

figure(1)
titol=[meth situ sstep]

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
        if st==0
            alpha=0.1;
        elseif st==1
            aux=svd(J2);
            alpha=1/max(aux);
            %alpha=max(min(aux),0.02);
        elseif st==2
            JJTe=J2*J2'*e2;
            alpha=dot(JJTe,e2)/dot(JJTe,JJTe);
        end
        
        %actualization
        %q=pi2pi(q+alpha*J2'*e2);
        q=q+alpha*J2'*e2;
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
        q=(q+alpha*Ju*e2)
        %q=(q+alpha*Ju*e2);
        %it
        
    elseif a==2
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
        q=pi2pi(q+dq);
        
    elseif a==3
        
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
        
        %actualization
        %dH=Hderiv(q,Qlim);
        %dq=alpha*(Ju*e2+mu*(eye(4)-Ju*J)*dH');
        q=pi2pi(q+dq);
        
        
    elseif a==4
        
        %FILTERED JACOBIAN
        %method
        it=it+1;
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J=jacob0(r,q);
        [UU,SS,VV]=svd(J);
        lambda2=sqrt(lambdamax^2*(1-(SS(n,n)/eps)^2));
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
        
        dq=0;
        for i=1:n-1
            SS2(i,i)=1/SS(i,i);
        end
        
        SS2(n,n)=SS(n,n)/(SS(n,n)^2+lambda^2);
        
        for i=1:n
            Ju=Ju+VV(:,i)*UU(:,i)'*SS2(i,i);
        end
        dq=alpha*Ju*e2;
        
        %actualization
        %dH=Hderiv(q,Qlim);
        %        dq=alpha*(Ju*e2+eye(4)*Ju*J)*dH;
        q=pi2pi(q+dq);
        
    elseif a==5
        
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
        
        q=pi2pi(q+dq);
        
        
        
    elseif a==6
        
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
        
        dq=H*Ju*e;
        
        %actualization
        
        q=pi2piD(q+dq,Qlim);
        
        
    elseif a==7
        
        %JOINT CLAMPING with continuous activation
        %method
        beta=0.1;
        H=HjlCONT(q,Qlim,beta);
        
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
        
        dq=H*Ju*e;
        
        %actualization
        
        q=pi2piD(q+dq,Qlim);
        %q=pi2pi(q+dq);
        
        
    elseif a==8
        
        %Jacobian continuous filtering
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
        Jb=UU*SS2*VV';
        
        
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
        
        %actualization
        %dH=Hderiv(q,Qlim);
        %        dq=alpha*(Ju*e2+eye(4)*Ju*J)*dH;
        q=pi2pi(q+dq);
        
        
    elseif a==9
        
        %SELECTIVELY DAMPED
        %method
        
        smin=0.01;
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
        q=pi2pi(q+dq);
        
        
    elseif a==10
        
        %TASK PRIORITY CLAMPING
        %method
        lambdajl=0.2;
        beta=0.1;
        it=it+1;
        J1=eye(7)-HjlCONT(q,Qlim,beta);
        e1=-lambdajl*q;
        q1=pinv(J1)*e1;
        P1=eye(7)-pinv(J1)*J1;
        
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J2=jacob0(r,q);
        
        
        q2=q1+pinv(J2*P1)*(e2+J2*q1);
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
        q=pi2pi(q+dq);
        
        
    elseif a==11
        
        %CONTINUOUS TASK PRIORITY CLAMPING
        %method
        lambdajl=0.2;
        
        beta=0.2;
        it=it+1;
        J1=eye(7);
        H=eye(7)-HjlCONT(q,Qlim,beta);
        e1=-lambdajl*q;
        q1=rcpinv(J1,H)*e1;
        P1=eye(7)-rcpinv(J1,H)*J1;
        %P1=eye(4)-pinv(J1)*J1;
        
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J2=jacob0(r,q);
        
        
        
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
        q=pi2piD(q+dq,Qlim);
        
        
    elseif a==12
        
        %CONTINUOUS TASK PRIORITY CLAMPING and smooth activation
        %method
        lambdajl=0.2;
        smin=0.01;
        beta=0.2;
        it=it+1;
        J1=eye(7);
        H=eye(7)-HjlCONT(q,Qlim,beta);
        e1=-lambdajl*q;
        q1=rcpinv(J1,H)*e1;
        P1=eye(7)-rcpinv(J1,H)*J1;
        %P1=eye(4)-pinv(J1)*J1;
        
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        J2=jacob0(r,q);
        
        [UU,SS,VV]=svd(J2);
        Ju=zeros(size(J2))';
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
        
        
        
        if norm(q2)>1
            hoa=1;
        else
        end
        
        dq=alpha*q2;
        %actualization
        %dH=Hderiv(q,Qlim);
        %       dq=alpha*(Ju*e2+eye(4)*Ju*J)*dH;
        q=pi2pi(q+dq);
        
    elseif a==15
        lambdajl=0.2;
        smin=0.01;
        beta=0.1;
        it=it+1;
        J1=eye(7);
        H=eye(7)-HjlCONT(q,Qlim,beta);
        e1=-lambdajl*q;
        q1=rcpinv(J1,H)*e1;
        P1=eye(7)-rcpinv(J1,H)*J1;
        %P1=eye(4)-pinv(J1)*J1;
        
        ex=xgd-xg;
        eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
        e2=[ex;eo];
        
        J=jacob0(r,q);
        [UU,SS,VV]=svd(J);
        Ju=zeros(size(J))';
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
        
        
        
        %--------------------------------------------------------------------
        %----------------------------------------------------------
    end
    
    
    T=fkine(r,q);
    aux=tr2eul(T);
    %xg=[T(1:2,4);pi2pi(aux(3))];
    xg=[T(1:3,4)];
    
    ex=xgd-xg;
    eo=0.5*(cross(T(1:3,1),Td(1:3,1))+cross(T(1:3,2),Td(1:3,2))+cross(T(1:3,3),Td(1:3,3)));
    e=[ex;eo];
    
    if or(norm(e)<emax,it>itmax)
        final=1;
    end
    %    plot(xgd(1),xgd(2),'*r');
    %    hold on
    %    plotroboprova(q,(it-1)/itmax)
    
    XX=[XX;xg'];
    QQ=[QQ;q'];
    EE=[EE;ex' eo' norm(e)];
    AA=[AA;alpha];
end


%  SHOW
qfinal=q
efinal=e
iterations=it
time=-etime(t0,clock)



%% PLOTS




%TRAJECTORY
figure(1)
subplot(3,2,1)
title 'Position trajectory'
plot3(xgd(1),xgd(2),xgd(3),'*r');
hold on
plot3(xgi(1),xgi(2),xgi(3),'*m');
for it2=1:it+1
    %       [P0,P1,P2,P3,P4,P5]=plotWAM(q,qlor,L)
    [Pu0,Pu1,Pu2,Pu3,Pu4,Pu5]=plotWAM(QQ(it2,:)',[1-(it2-1)/it,1-(it2-1)/it,(it2-1)/it],1);
end

xlabel 'x'
ylabel 'y'
zlabel 'z'
title '3D Trajectory'

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
