function out = ncalculest(robot,model1,model2,model3,model4,model5,model6,u)
n=robot.n;
q=u(1:n);
dq=u(n+1:2*n);
ddq=u(2*n+1:3*n);
a=0;

if a==1
    tau(1)=lwpr_predict(model1,u,0.001);
    tau(2)=lwpr_predict(model2,u,0.001);
    tau(3)=lwpr_predict(model3,u,0.001);
    tau(4)=lwpr_predict(model4,u,0.001);
    tau(5)=lwpr_predict(model5,u,0.001);
    tau(6)=lwpr_predict(model6,u,0.001);
    out1=(tau'-inertia(robot,q')*ddq);
    out=out1;
else 

    out2=coriolis(robot,q',dq')'+gravload(robot,q')';
    out=out2;
end


