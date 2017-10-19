d = 16;
n = 10;

x = randn (d, n);
%posnegx = find(x<0);
%x(posnegx)=-x(posnegx);

C = ones(1,n) / sqrt(n);


% Sum
[alpha1, y1, x1] = qdemocratic (x, 'sum', 0);

Lx1 = sum(x1'*x1, 2);
Lx1 = Lx1 ./ norm(Lx1);
C * Lx1


% Sinkhorn
[alpha2, y2, x2] = qdemocratic (x, 'sinksp', 0);

Lx2 = sum(x2'*x2, 2);
Lx2 = Lx2 ./ norm(Lx2);
C * Lx2


% ZCA
[alpha3, y3, x3, z] = qdemocratic (x, 'zca', 0);

Lx3 = sum(x3'*x3, 2);
Lx3 = Lx3 ./ norm(Lx3);
C * Lx3

% ZCAw
[alpha4, y4, x4] = qdemocratic (x, 'zcaw', 0);

Lx4 = sum(x4'*x4, 2);
Lx4 = Lx4 ./ norm(Lx4);
C * Lx4
