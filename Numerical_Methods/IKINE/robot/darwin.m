clear L
%alpha a theta d
L{1} = link([-pi/2 -16 0 21.65 0],'standard');
L{2} = link([pi/2 -60 0 16 0],'standard');
L{3} = link([0 -130 0 0 0],'standard');
%L{4} = link([-pi/2 0 pi/2 0 0],'standard');
L{4} = link([0 -50 0 0 0],'standard');
qoff=[0 0 0 pi/2];
darwin = robot(L);