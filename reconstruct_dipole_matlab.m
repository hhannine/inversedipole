% Include dependencies: AIRToolsII, Regtools
addpath(genpath('./dependencies'))

clear all

% % xbj=0.002
% load('./exports/exp_fwdop_simulated-lo-sigmar_MVgamma_dipole-lightonly_newbins_r_steps200_xbj0.002.mat')
% load('./exports/exp_fwdop_heraII_filtered_s318.1_xbj0.002_lightonly_r_steps200.mat')
% load('./data/reconstruction_help/sigr0.002.mat')
% load('./data/reconstruction_help/q2vals0.002.mat')


%xbj=0.008
load('./exports/exp_fwdop_simulated-lo-sigmar_MVgamma_dipole-lightpluscharm_newbins_r_steps200_xbj0.008.mat')
load('./exports/exp_fwdop_heraII_filtered_s318.1_xbj0.008_lightpluscharm_r_steps200.mat')
load('./data/reconstruction_help/sigr0.008.mat')
load('./data/reconstruction_help/q2vals0.002.mat')

%xbj=0.005 (Ntrue for xbj=0.001)
% load('./exports/export_discrete_operator_simulated-lo-sigmar_MVgamma_dipole0.002.mat')
% load('./exports/export_discrete_operator_heraII_filtered_s318.1_xbj0.005.mat')
% load('sigr0.005.mat')

%xbj=0.008 (Ntrue for xbj=0.001)
% load('./exports/export_discrete_operator_simulated-lo-sigmar_MVgamma_dipole0.01.mat')
% load('./exports/export_discrete_operator_heraII_filtered_s318.1_xbj0.008.mat')
% load('sigr0.008.mat')


%xbj=0.0032 (Ntrue for xbj=0.001)
% load('./exports/export_discrete_operator_simulated-lo-sigmar_MVgamma_dipole0.002.mat')
% load('./exports/export_discrete_operator_heraII_filtered_s318.1_xbj0.0032.mat')
% load('sigr0.0032.mat')
% load('q2vals0.0032.mat')
%%


% ivec = [1 150 300 450 600 700 720 740 760 780 800 820 840 860 880 900 925 950 975 1000];
% ivec2=[1:100:500, 501:10:1000];
% ivec3=[1:800, 801:5:1000];

% % ivec3= 1:5:1000;
% ivec3= 1:1000;
ivec3= 1:200;



x = discrete_dipole_N;
A = forward_op_A(:,ivec3);
% bex = A*x';
x = x(ivec3);
bfit = A*x';

b = sigr;

% rng(80,"twister");
% eta = 0.01;
% e = randn(size(bex));
% e = eta*norm(bex)*e/norm(e);
% b = bex + e;


%%
N=length(x);
[L1,W1]=get_l(N,1);
[L2,W2]=get_l(N,2);

[U,s,V] = csvd(A);

[UU,sm,XX] = cgsvd(A,L1);

[UU2,sm2,XX2] = cgsvd(A,L2);


%%
lambda = [1,3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,3e-5];


X_tikh = tikhonov(UU,sm,XX,b,lambda);

errtik = zeros(size(lambda));


for i = 1:length(lambda)
    errtik(i) = norm((x'-X_tikh(:,i)))/norm(x');
end

[m,mI]=min(errtik);

figure(1)
plot(ivec3,x','-',ivec3,X_tikh(:,mI),'--','LineWidth',3)
leg=legend('true','PTik')
set(leg,'FontSize',14);
% title('Preconditioned Tikhonov with xbj = 0.002','FontSize',16)


title('Preconditioned Tikhonov with xbj = 0.002','FontSize',16)

%%
lambda = linspace(1,10,10);%[1,3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,3e-5];


X_tikh = tikhonov(UU,sm,XX,b,lambda);

errtik = zeros(size(lambda));


for i = 1:length(lambda)
    errtik(i) = norm((x'-X_tikh(:,i)))/norm(x');
end

[m,mI]=min(errtik);


figure(2)
plot(ivec3,x','-',ivec3,X_tikh(:,mI),'--','LineWidth',3)
leg=legend('true','PTik')
set(leg,'FontSize',14);
% title('Preconditioned Tikhonov with xbj = 0.002','FontSize',16)


title('Preconditioned Tikhonov with xbj = 0.002','FontSize',16)

%%
figure(8)
plot(q2vals,b,'o',q2vals,bfit,'*')

%%
figure(6)
for i = 1:length(lambda)
plot(1:N,x','-',1:N,X_tikh(:,i),'--','LineWidth',3)
ylim([-0.4,1.1])
hold on
end
hold off

%%

bend = A*X_tikh(:,mI);

figure(7)
plot(q2vals,b,'bo-',q2vals,bfit,'ro-',q2vals,bend,'go-','LineWidth',2)
legend('b - measurements','b - fit','b - reconstruction')

title('How well does the reconstruction fit the data for xbj = 0.002','FontSize',16)

%%

% 
% rng(80,"twister");
% eta = 0.01;
% e = randn(size(bfit));
% e = eta*norm(bfit)*e/norm(e);
% b1noise = bfit + e;
% 
% %%
% 
% figure(8)
% plot(q2vals,b,'bo-',q2vals,bfit,'ro-',q2vals,b1noise,'go-','LineWidth',2)


