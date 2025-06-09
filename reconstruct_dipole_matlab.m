% Include dependencies: AIRToolsII, Regtools
addpath(genpath('./dependencies'))

close all
clear all

%         1       2        3        4            5
fits = ["MV_", "MVgamma", "MVe", "bayesMV4", "bayesMV5"];
fitname = fits(4);

all_xbj_bins = [1e-05, 0.0001, 0.00013, 0.0002, 0.00032, 0.0005, 0.0008, 0.001, 0.0013, 0.002, 0.0032, 0.005, 0.008, 0.01];
% xbj_bin = "1e-05";
real_xbj_bins = [0.00013, 0.0002, 0.00032, 0.0005, 0.0008, 0.0013, 0.002, 0.0032, 0.005, 0.008, 0.013, 0.02, 0.032, 0.05, 0.08];

r_steps = 500;
r_steps_str = strcat("r_steps",int2str(r_steps));
% use_real_data = false;
use_real_data = true;
use_charm = false;
% use_charm = true;
if use_real_data
    all_xbj_bins = real_xbj_bins;
end

% lambda_type = "broad";
% lambda_type = "semiconstrained";
% lambda_type = "semicon2";
lambda_type = "fixed";
% lambda_type = "semifix";
% lambda_type = "old";
if lambda_type == "broad"
    lam1 = 1:9;
    lambda = [lam1*1e-7, lam1*1e-6, lam1*1e-5, lam1*1e-4, lam1*1e-3, lam1*1e-2];
elseif lambda_type == "semiconstrained"
    lambda = [0.01, 0.02, 0.03, 0.04, 0.05]; % semi-constrained
elseif lambda_type == "semicon2"
    lambda = [0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09];
elseif lambda_type == "semifix"
    lambda = [0.01, 0.02, 0.03, 0.04, 0.05];
elseif lambda_type =="fixed"
    % lambda = [0.01];
    fac = 1.2;
    lambda = [0.01*fac^-2, 0.01*fac^-1, 0.01*fac^0, 0.01*fac^1, 0.01*fac^2];
elseif lambda_type == "old"
    lambda = [5e-1,4e-1,3e-1,1e-1,9e-2,8e-2,7e-2,5e-2,3e-2,1e-2,9e-3,7e-3,5e-3,3e-3,1e-3,8e-4,4e-4,1e-4,8e-5,4e-5,2e-5,1e-5];
else
    "BAD LAMBDA TYPE"
end

charm_opt = "lightonly"; % new files omitted this unfortunately
if (use_charm)
    charm_opt = "lightpluscharm";
end
sim_charm_opt = charm_opt;
data_type = fitname;
if (use_real_data)
    data_type = "heraII_filtered";
end

% load forward operator file
% data_path = './exports/';
% data_path = './exports_unitysigma/';
data_path = './export_fwd_IUSinterp/';
data_files = dir(fullfile(data_path,'*.mat'));
sim_type = "simulated";

for xi = 1:length(all_xbj_bins)
    close all
    xbj_bin = string(all_xbj_bins(xi));
    % [fitname, xbj_bin, r_steps,use_real_data,use_charm]

    for k = 1:numel(data_files)
        fname = data_files(k).name;
        if (contains(fname, xbj_bin) && contains(fname, data_type) && contains(fname, charm_opt) && contains(fname, r_steps_str))
            run_file = fname
        end
    end
    load(strcat(data_path, run_file))
    
    % if using real data, need to load reference fit dipole separately
    dip_file="";
    for k = 1:numel(data_files)
        if str2double(xbj_bin) > 0.01
            sim_xbj = '0.01';
        else
            sim_xbj = xbj_bin;
        end
        fname = data_files(k).name;
        fnameb = strrep(fname,'-','_');
        if (contains(fnameb, sim_xbj) && contains(fnameb, fitname) && contains(fnameb, sim_type) && contains(fnameb, charm_opt) && contains(fnameb, r_steps_str))
            dip_file = fname;
            break
        end
    end
    if dip_file == ""
            ["failed to match dip_file!", fname, sim_xbj, fitname, sim_type, charm_opt, r_steps_str]
            return
    end
    dip_file
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
    x = real_sigma*x(ivec3);
    bfit = A*x'; % bfit has numerical error from discretization
    % b is either the real data sigma_r, or one simulated by fit
    b = sigmar_vals'; % b is calculated by the C++ code, no error.
    % todo bfit_errs??? Maybe we can say that they're apples and oranges,
    % not direcly comparable?

    b_hera = sigmar_vals';
    b_errs = sigmar_errs'; % THIS IS NEEDED TO DO THE DATA \pm error reconstructions!
    % only do best reconst to b_err_upper and b_err_lower
    b_err_upper = b_hera + b_errs;
    b_err_lower = b_hera - b_errs;
    bfit_plus_err = bfit + b_errs;
    bfit_minus_err = bfit - b_errs;


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
    %lambda

    X_tikh = tikhonov(UU,sm,XX,b,lambda);
    errtik = zeros(size(lambda));

    X_tikh_upper = tikhonov(UU,sm,XX,b_err_upper,lambda);
    % errtik_upper = zeros(size(lambda));
    X_tikh_lower = tikhonov(UU,sm,XX,b_err_lower,lambda);
    % errtik_lower = zeros(size(lambda));
    
    for i = 1:length(lambda)
        % errtik(i) = norm((x'-X_tikh(:,i)))/norm(x');
        % errtik(i) = norm((b-A*X_tikh(:,i)))/norm(b); % this is wrong, cannot really minimize against b since it has error
        errtik(i) = abs(norm((b-A*X_tikh(:,i))/(b_errs))-1); 
    end
    [m,mI]=min(errtik);
    % if lambda_type == "semifix"
    %     mI = 3;
    % else
    %     [m,mI]=min(errtik);
    % end
    best_lambda = lambda(mI);
    [mI, best_lambda, lambda_type]

    N_maxima = [];
    N_bpluseps_maxima = [];
    N_bminuseps_maxima = [];
    if (length(lambda)>5) && (mI>=3) && (mI<=length(lambda)-2)
        for i = 1:5
            N_maxima(i) = max(X_tikh(:,mI-3+i));
        end
    else
        for i = 1:length(lambda)
            N_maxima(i) = max(X_tikh(:,i));
        end
    end
    % N_maxima
    % [xbj_bin, N_maxima(1), lambda(mI)]
    % real_sigma
    
    figure(1) % best reconstruction vs. ground truth
    % plot(ivec3,x','-',ivec3,X_tikh(:,mI),'--','LineWidth',2)
    r_grid(end) = []; 
    % plot(r_grid',x','-',r_grid',X_tikh(:,mI),'--','LineWidth',2)
    semilogx(r_grid',x','-',r_grid',X_tikh(:,mI),'--','LineWidth',2)
    xlim([r_grid(1), r_grid(end)]);
    leg=legend('true',['PTik lambda=', num2str(lambda(mI))]);
    set(leg,'FontSize',12);
    set(leg,'Location',"northwest");
    title(['Preconditioned Tikhonov with xbj=',num2str(xbj_bin)],'FontSize',12)
    pos1 = get(gcf,'Position'); % get position of Figure(1) 
    % set(gcf,'Position', pos1 - [pos1(3)/2,0,0,0]) % Shift position of Figure(1) 
    
    %% Comparing lambdas
    figure(2)
    semilogx(r_grid',x','-','LineWidth',1)
    hold on
    % for i = 1:length(lambda)
    % mI
    % length(lambda)
    if mI<3 | length(lambda)-mI <3
        for i = 1:length(lambda)
            % plot(1:N,x','-',1:N,X_tikh(:,i),'--','LineWidth',1)
            semilogx(r_grid',X_tikh(:,i),'--','LineWidth',1)
            % ylim([-0.4,1.5]);
            xlim([0.1, r_grid(end)]);
            hold on
        end
        legend_lambdas = cellstr(num2str(lambda', 'lambda=%-f'));
        legend_items = cat(1, 'true', legend_lambdas(1:length(lambda)));
    else
        for i = mI-2:mI+2
            % plot(1:N,x','-',1:N,X_tikh(:,i),'--','LineWidth',1)
            semilogx(r_grid',X_tikh(:,i),'--','LineWidth',1)
            % ylim([-0.4,1.5]);
            xlim([0.1, r_grid(end)]);
            hold on
        end
        legend_lambdas = cellstr(num2str(lambda', 'lambda=%-f'));
        legend_items = cat(1, 'true', legend_lambdas(mI-2:mI+2));
    end
    title(['Reconstruction with various lambda xbj=',num2str(xbj_bin)],'FontSize',12)
    
    leg=legend(legend_items);
    set(leg,'FontSize',10);
    set(leg,'Location',"northwest");
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
    set(gcf,'Position', pos3 + [2/2*pos2(3),0,0,0]) % Shift position of Figure(2)
    
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
    
    if use_real_data
        reconst_type = "data_only";
    end
    if (use_real_data==false)
        if (contains(run_file, "_MV_"))
            reconst_type = "MV";
        elseif (contains(run_file, "_MVe_"))
            reconst_type = "MVe";
        elseif (contains(run_file, "_MVgamma_"))
            reconst_type = "MVgamma";
        elseif (contains(run_file, "_bayesMV4_"))
            reconst_type = "bayesMV4";
        elseif (contains(run_file, "_bayesMV5_"))
            reconst_type = "bayesMV5";
        else
            reconst_type = "FITNAME_NOT_RECOGNIZED";
            error([reconst_type ' with ' run_file]);
        end
    end
    if (use_real_data)
        data_name = "hera";
    else
        data_name = "sim";
    end
    if (use_charm)
        flavor_string = "lightpluscharm";
    else
        flavor_string = "lightonly";
    end
    name = [data_name, '_', reconst_type, '_', flavor_string, '_', lambda_type];
    recon_path = "./reconstructions_IUSdip/";
    f_exp_reconst = strjoin([recon_path 'recon_out_' name '_xbj' xbj_bin '.mat'],"")
    N_reconst = X_tikh(:,mI);
    N_rec_adjacent = X_tikh;
    N_reconst_from_b_plus_err = X_tikh_upper(:,mI);
    N_bpluseps_rec_adjacent = [];
    N_reconst_from_b_minus_err = X_tikh_lower(:,mI);
    N_bminuseps_rec_adjacent = [];
    N_fit = discrete_dipole_N;
    b_cpp_sim = b; % data generated in C++, no discretization error.
    b_fit = bfit; % = A*Nfit, this has discretization error.
    b_from_reconst = bend; % prescription of the data by the reconstructred dipole.
    b_from_reconst_adjacent = [];
    for i = 1:length(lambda)
        b_from_reconst_adjacent(:,i) = A*X_tikh(:,i);
    end
    b_plus_err_from_reconst = A*X_tikh_upper(:,mI); % should we include lambda variation here also? No?
    b_minus_err_from_reconst = A*X_tikh_lower(:,mI); % should we include lambda variation here also? No?
    save(f_exp_reconst, ...
        "r_grid", "q2vals", ...
        "N_fit", "real_sigma",...
        "N_reconst", "N_maxima", "N_rec_adjacent", ...
        "N_reconst_from_b_plus_err", "N_bpluseps_maxima", "N_bpluseps_rec_adjacent", ...
        "N_reconst_from_b_minus_err", "N_bminuseps_maxima", "N_bminuseps_rec_adjacent", ...
        "b_cpp_sim", "b_fit", "b_from_reconst", "b_from_reconst_adjacent", ...
        "b_hera", "b_errs", ...
        "b_plus_err_from_reconst", "b_minus_err_from_reconst", ...
        "best_lambda", "lambda", "lambda_type", ...
        "xbj_bin", "use_real_data", "use_charm", ...
        "run_file", "dip_file", ...
        "-nocompression","-v7")
end

%% with real data, print max / peak and xbj_bin to a file