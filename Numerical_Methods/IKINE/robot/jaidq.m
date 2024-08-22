function y=jaidq(robot, u)
n=robot.n;
q=u(1:n);
dq=u(n+1:2*n);
Te=fkine(robot,q);
eul=tr2eul(Te);
Xe=[Te(1:3,4);eul'];
Taux=[0 -sin(eul(1)) cos(eul(1))*sin(eul(2));0 cos(eul(1)) cos(eul(1))*sin(eul(2));1 0 cos(eul(2))];
Ti=[eye(3) zeros(3); zeros(3) pinv(Taux)];
J=jacobn(robot,q);
Ja=Ti*J;

y=pinv(Ja)*dq;