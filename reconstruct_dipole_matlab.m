% Include dependencies: AIRToolsII, Regtools
addpath(genpath('./dependencies'))

close all
clear all

%         1       2        3        4            5
fits = ["MV_", "MVgamma", "MVe", "bayesMV4", "bayesMV5"];
fitname = fits(5);

all_xbj_bins = [1e-05, 0.0001, 0.00013, 0.0002, 0.00032, 0.0005, 0.0008, 0.001, 0.0013, 0.002, 0.0032, 0.005, 0.008, 0.01];
% xbj_bin = "0.0001"
% xbj_bin = "1e-05";
xbj_bin = "0.001";
% xbj_bin = "0.01";
r_steps = 500;
r_steps_str = strcat("r_steps",int2str(r_steps));
use_real_data = false;
% use_real_data = true;
% use_charm = false;
use_charm = true;
[fitname, xbj_bin, r_steps,use_real_data,use_charm]

% charm_opt = "lightonly"; % new files omitted this unfortunately
charm_opt = "dipole_r";
if (use_charm)
    charm_opt = "lightpluscharm";
end
data_type = fitname;
if (use_real_data)
    data_type = "heraII_filtered";
end

% load forward operator file
data_path = './exports/';
data_files = dir(fullfile(data_path,'*.mat'));
for k = 1:numel(data_files)
    fname = data_files(k).name;
    if (contains(fname, xbj_bin) && contains(fname, data_type) && contains(fname, charm_opt) && contains(fname, r_steps_str))
        run_file = fname
    end
end
load(strcat(data_path, run_file))

% if using real data, need to load reference fit dipole separately
sim_type = "simulated";
dip_file = 'exp_fwdop+data_simulated-lo-sigmar_MVgamma_dipole-lightpluscharm_newbins_r_steps200_xbj0.0008.mat'; % backup file to have something
for k = 1:numel(data_files)
    fname = data_files(k).name;
    if (contains(fname, fitname) && contains(fname, xbj_bin) && contains(fname, sim_type) && contains(fname, charm_opt) && contains(fname, r_steps_str))
        dip_file = fname
    end
end
dip_data = load(strcat(data_path, dip_file));
ref_dip = dip_data.discrete_dipole_N;
if (use_real_data)
    discrete_dipole_N = ref_dip;
end
%%

ivec3= 1:r_steps;
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
% [U,s,V] = csvd(A);

% first order derivative operator
[UU,sm,XX] = cgsvd(A,L1);

% second order derivative operator
% [UU2,sm2,XX2] = cgsvd(A,L2);


%%
% lambda = [1,3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,3e-5];
% lambda = [6e-1,5e-1,4e-1,3e-1,1e-1,9e-2,8e-2,7e-2,5e-2,3e-2,1e-2,9e-3,7e-3,5e-3,3e-3,1e-3];
lambda = [1e-1,9e-2,8e-2,7e-2,5e-2,3e-2,1e-2,9e-3,7e-3,5e-3,3e-3];
X_tikh = tikhonov(UU,sm,XX,b,lambda);
errtik = zeros(size(lambda));

for i = 1:length(lambda)
    errtik(i) = norm((x'-X_tikh(:,i)))/norm(x');
end
[m,mI]=min(errtik);

figure(1) % best reconstruction vs. ground truth
% plot(ivec3,x','-',ivec3,X_tikh(:,mI),'--','LineWidth',2)
r_grid(end) = []; 
loglog(r_grid',x','-',r_grid',X_tikh(:,mI),'--','LineWidth',2)
leg=legend('true',['PTik lambda=', num2str(lambda(mI))]);
set(leg,'FontSize',12);
set(leg,'Location',"northwest");
title(['Preconditioned Tikhonov with xbj=',num2str(xbj_bin)],'FontSize',12)
pos1 = get(gcf,'Position'); % get position of Figure(1) 
set(gcf,'Position', pos1 - [pos1(3)/2,0,0,0]) % Shift position of Figure(1) 

%% Comparing lambdas
figure(2)
plot(1:N,x','-','LineWidth',1)
hold on
for i = 1:length(lambda)
% plot(1:N,x','-',1:N,X_tikh(:,i),'--','LineWidth',1)
plot(1:N,X_tikh(:,i),'--','LineWidth',1)
ylim([-0.4,1.5])
hold on
end
title(['Reconstruction with various lambda xbj=',num2str(xbj_bin)],'FontSize',12)
legend_lambdas = cellstr(num2str(lambda', 'l=%-f'));
legend_items = cat(1, 'true', legend_lambdas);
leg=legend(legend_items);
set(leg,'FontSize',10);
set(leg,'Location',"southeast");
hold off
pos2 = get(gcf,'Position');  % get position of Figure(2) 
set(gcf,'Position', pos2 + [pos1(3)/2,0,0,0]) % Shift position of Figure(2)

%% Plot: data b vs fit b vs b from reconstruction

q2vals = qsq_vals;
bend = A*X_tikh(:,mI);

figure(3)
% plot(q2vals,b,'bx-',q2vals,bfit,'ro',q2vals,bend,'go','LineWidth',1)
% semilogx(q2vals,b,'bx',q2vals,bfit,'ro',q2vals,bend,'go','LineWidth',0.4)
loglog(q2vals,b,'bx',q2vals,bfit,'ro',q2vals,bend,'go','LineWidth',0.4)
leg=legend('b - measurements','b - fit','b - reconstruction');
title(['How well does the reconstruction fit the data for xbj=',num2str(xbj_bin)],'FontSize',12)
set(leg,'FontSize',10);
set(leg,'Location',"southeast");
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


% EXPORTING RESULTS

%% export reconstructed dipole
% variables to export: N_reconst, N_true = N_fit, b_data = b_sim, b_from_reconst =
% A*N_reconst, r_grid, q2vals

% filename should have all settings / parameters
% num2str(a_value,'%.2f') ; formatting

fitname = "data_only";
if (use_real_data==false)
    if (contains(run_file, "_MV_"))
        fitname = "MV";
    elseif (contains(run_file, "_MVe_"))
        fitname = "MVe";
    elseif (contains(run_file, "_MVgamma_"))
        fitname = "MVgamma";
    elseif (contains(run_file, "_bayesMV4_"))
        fitname = "bayesMV4";
    elseif (contains(run_file, "_bayesMV5_"))
        fitname = "bayesMV5";
    else
        fitname = "FITNAME_NOT_RECOGNIZED";
        run_file
        error([fitname ' with ' run_file]);
    end
end
data_name = "sim";
if (use_real_data)
    data_name = "hera";
end
flavor_string = "lightonly";
if (use_charm)
    flavor_string = "lightpluscharm";
end
name = [data_name, '_', fitname, '_', flavor_string];
f_exp_reconst = strjoin(['recon_out_' name '_xbj' xbj_bin '.mat'],"")
N_reconst = X_tikh(:,mI);
N_fit = discrete_dipole_N;
b_cpp_sim = b; % data generated in C++, no discretization error.
b_fit = bfit; % = A*Nfit, this has discretization error.
b_from_reconst = bend; % prescription of the data by the reconstructred dipole.
save(f_exp_reconst, ...
    "r_grid", "q2vals", ...
    "N_reconst", "N_fit", ...
    "b_cpp_sim", "b_fit", "b_from_reconst", ...
    "xbj_bin", "use_real_data", "use_charm", ...
    "run_file", "dip_file", ...
    "-nocompression","-v7.3")

%% with real data, print max / peak and xbj_bin to a file