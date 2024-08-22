function [h,q]=clampjoints(q,Qlim)
h=eye(size(q,1));
for i=1:size(q,1)
   if or(q(i)>Qlim(i,2),q(i)<Qlim(i,1))
       h(i,i)=0;
       q(i)=max(Qlim(i,1),min(Qlim(i,2),q(i)));
   end
      
    
end