function y = apply_H(p,A,At,Lr,Lrt,Lx,Lxt,alphar,alphax)

Ap = A*p;

term1 = At*Ap;

term2 = 2*alphar*(Lrt*(Lr*p));

term3 = 2*alphax*(Lxt*(Lx*p));

y = term1 + term2 + term3;

end