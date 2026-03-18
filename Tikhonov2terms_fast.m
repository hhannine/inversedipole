function x = Tikhonov2terms_fast(m,A,Lr,Lx,alphar,alphax,MAXITER,tol)

[~,c] = size(A);

rho = zeros(MAXITER,1);

At = A';
Lrt = Lr';
Lxt = Lx';

b = At*m;
bnorm = norm(b);

x = zeros(c,1);

% compute initial residual
Hx = apply_H(x,A,At,Lr,Lrt,Lx,Lxt,alphar,alphax);
r = b - Hx;

rho(1) = r'*r;
p = r;

k = 1;

while k < MAXITER

    w = apply_H(p,A,At,Lr,Lrt,Lx,Lxt,alphar,alphax);

    alpha = rho(k) / (p'*w);

    x = x + alpha*p;
    r = r - alpha*w;

    rho(k+1) = r'*r;

    relres = sqrt(rho(k+1))/bnorm;
    if relres < tol
        return
    end

    beta = rho(k+1)/rho(k);
    p = r + beta*p;

    k = k+1;

end

end