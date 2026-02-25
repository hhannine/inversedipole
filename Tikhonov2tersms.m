function x = Tikhonov2tersms(m,A,Lr,Lx,alphar,alphax,MAXITER,tol)
% Implementation of conjugate gradient method for deconvolution. We consider the model
%
%   m = Ax + epsilon,
%
% where m is measured image, x is ideal image, A is a convolution matrix with system PSF 
% as kernel and epsilon is pixel noise with Gaussian distribution (uniform variance).
% To solve the problem with statistical inversion, we minimize
%
%   f(x) = -x^T b + 1/2* x^T H x,
%
% where b = A^T m = Am and
%
%   H = 1/2* A^T A + alpha* Dv^T Dv + alpha* Dh^T Dh.   
%
% Here Dh applied to an image computes differences of horizontally adjacent pixels and Dv
% differences of vertically adjacent pixels. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This file implements the deconvolution using conjugate gradients. 
%
% Arguments:
% m         the noisy image (size of m is [row,col])
% A         forward operator
% Lr        L matrix in r-direction
% Lx        L matrix in xbj-direction
% alphar    regularization parameter corresponding to r-direction (positive real number)
% alphax    regularization parameter corresponding to xbj-direction (positive real number)
% MAXITER   maximum number of iterations
%
% Returns:
% x        denoised image
%
%
% Samuli Siltanen May 2001 adapted by Hjørdis Schlüter 2026
[rrr,ccc] = size(m);

[r,c] = size(A);

% Initializations
rho     = zeros(1,MAXITER);
b       = (A')*m;
bnorm   = sum(b(:)).^2;
x       = zeros(c,1);

Hx = (A')*A*x + 2*alphar*(Lr')*Lr*x + 2*alphax*(Lx')*Lx*x;

% Step 1.
r      = b - Hx;
rho(1) = sum(r(:).^2);
k      = 1;
disp(['Initial relative residual = ', num2str(sqrt(rho(1))/bnorm)])

% Step 2.
while k < MAXITER
    %(a)
    if k==1
        p = r;
    else
        beta = rho(k)/rho(k-1);
        p    = r + beta*p;
    end
    w        = (A')*A*p + 2*alphar*(Lr')*Lr*p + 2*alphax*(Lx')*Lx*p;
    a        = rho(k) / (p(:).'*w(:));
    x        = x + a*p;
    r        = r - a*w;
    rho(k+1) = sum(r(:).^2);
    k        = k+1;
    %disp(['Residual after iteration ', num2str(k), ' is ', num2str(sqrt(rho(k))/bnorm)])
    relres = sqrt(rho(k))/bnorm;
    if relres < tol
        %disp(k)
        return
    end
end
disp(['Residual after iteration ', num2str(k), ' is ', num2str(sqrt(rho(k))/bnorm)])
end


