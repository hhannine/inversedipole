% Include dependencies: AIRToolsII, Regtools
addpath(genpath('./dependencies'))

close all
clear all

xbj_bin = "0.0032";
use_real_data = true;
use_charm = true;

charm_opt = "lightonly";
if (use_charm)
    charm_opt = "lightpluscharm";
end
data_type = "simulated";
if (use_real_data)
    data_type = "heraII_filtered";
end

% load forward operator file
data_path = './exports/';
data_files = dir(fullfile(data_path,'*.mat'));
for k = 1:numel(data_files)
    fname = data_files(k).name;
    if (contains(fname, xbj_bin) && contains(fname, data_type) && contains(fname, charm_opt))
        run_file = fname;
    end
end
load(strcat(data_path, run_file))

% if using real data, need to load reference fit dipole separately
sim_type = "simulated";
for k = 1:numel(data_files)
    fname = data_files(k).name;
    if (contains(fname, xbj_bin) && contains(fname, sim_type) && contains(fname, charm_opt))
        dip_file = fname;
    end
end
dip_data = load(strcat(data_path, dip_file));
ref_dip = dip_data.discrete_dipole_N;
if (use_real_data)
    discrete_dipole_N = ref_dip;
end

% % xbj=0.002 lightonly
% load('./exports/exp_fwdop_simulated-lo-sigmar_MVgamma_dipole-lightonly_newbins_r_steps200_xbj0.002.mat')
% load('./exports/exp_fwdop_heraII_filtered_s318.1_xbj0.002_lightonly_r_steps200.mat')
% load('./data/reconstruction_help/sigr0.002.mat')
% load('./data/reconstruction_help/q2vals0.002.mat')

% % xbj=0.002 lightpluscharm
% load('./exports/exp_fwdop_simulated-lo-sigmar_MVgamma_dipole-lightpluscharm_newbins_r_steps200_xbj0.002.mat')
% load('./exports/exp_fwdop_heraII_filtered_s318.1_xbj0.002_lightpluscharm_r_steps200.mat')
% load('./data/reconstruction_help/sigr0.002.mat')
% load('./data/reconstruction_help/q2vals0.002.mat')

%%

ivec3= 1:200;
% ivec3 = 1:20:200;

x = discrete_dipole_N;
A = forward_op_A(:,ivec3);
% bex = A*x';
x = x(ivec3);
bfit = A*x'; % bfit has numerical error from discretization

% b is either the real data sigma_r, or one simulated by fit
b = sigmar_vals'; % b is calculated by the C++ code, no error.

% rng(80,"twister");
% eta = 0.01;
% e = randn(size(bex));
% e = eta*norm(bex)*e/norm(e);
% b = bex + e;


%%
N=length(x);
[L1,W1]=get_l(N,1);
[L2,W2]=get_l(N,2);

% classical 0th order tikhonov
[U,s,V] = csvd(A);

% first order derivative operator
[UU,sm,XX] = cgsvd(A,L1);

% second order derivative operator
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
title('Preconditioned Tikhonov with xbj = 0.002','FontSize',16)
pos1 = get(gcf,'Position'); % get position of Figure(1) 
set(gcf,'Position', pos1 - [pos1(3)/2,0,0,0]) % Shift position of Figure(1) 

%% Comparing lambdas
figure(2)
for i = 1:length(lambda)
plot(1:N,x','-',1:N,X_tikh(:,i),'--','LineWidth',3)
ylim([-0.4,1.5])
hold on
end
title('Reconstructions at various lambda, xbj = 0.002','FontSize',16)
hold off
pos2 = get(gcf,'Position');  % get position of Figure(2) 
set(gcf,'Position', pos2 + [pos1(3)/2,0,0,0]) % Shift position of Figure(2)

%% Plot: data b vs fit b vs b from reconstruction

q2vals = qsq_vals;
bend = A*X_tikh(:,mI);

figure(3)
plot(q2vals,b,'bx',q2vals,bfit,'ro',q2vals,bend,'go','LineWidth',1)
legend('b - measurements','b - fit','b - reconstruction')
title('How well does the reconstruction fit the data for xbj = 0.002','FontSize',12)
pos3 = get(gcf,'Position');  % get position of Figure(3) 
set(gcf,'Position', pos3 + [3*pos2(3)/2,0,0,0]) % Shift position of Figure(2)

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


