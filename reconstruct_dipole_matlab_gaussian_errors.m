% Include dependencies: AIRToolsII, Regtools
addpath(genpath('./dependencies'))

close all
clear all

% parpool(8)
parp = gcp;

all_xbj_bins = [1e-05, 0.0001, 0.00013, 0.0002, 0.00032, 0.0005, 0.0008, 0.001, 0.0013, 0.002, 0.0032, 0.005, 0.008, 0.01];
real_xbj_bins = [0.00013, 0.0002, 0.00032, 0.0005, 0.0008, 0.0013, 0.002, 0.0032, 0.005, 0.008, 0.013, 0.02, 0.032, 0.05, 0.08];

r_steps = 500;
r_steps_str = strcat("r_steps",int2str(r_steps));


%%% Fit options
%         1       2        3        4            5
fits = ["MV_", "MVgamma", "MVe", "bayesMV4", "bayesMV5"];
fitname = fits(4);

%%% simulated data settings
use_real_data = false;
use_charm = false;
% use_charm = true;

%%% real data settings
% use_real_data = true; TODO NEED TO REDO THE ERROR STUFF FOR REAL ERRORS
% use_charm = false;
% use_charm = true;

%%% Lambda options for production
% lambda_type = "fixed";
lambda_type = "broad"; % for simulated data


%%% All lambda options
% lambda_type = "broad";
% lambda_type = "semiconstrained";
% lambda_type = "semicon2";
% lambda_type = "fixed";
% lambda_type = "semifix";
% lambda_type = "old";

if lambda_type == "broad"
    lam1 = 1:9;
    % lam1 = 2:2:10;
    % lambda = [lam1*1e-7, lam1*1e-6, lam1*1e-5, lam1*1e-4, lam1*1e-3, lam1*1e-2];
    % lambda = [lam1*1e-6, lam1*1e-5, lam1*1e-4, lam1*1e-3, lam1*1e-2];
    % lambda = [lam1*1e-5, lam1*1e-4, lam1*1e-3, lam1*1e-2];
    lambda = [lam1*1e-4, lam1*1e-3, lam1*1e-2]; % This is quite good and wide for 1st order Tikh!
    % lambda = [lam1*4e-3, lam1*1e-2];
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

if use_real_data
    all_xbj_bins = real_xbj_bins;
end

% load forward operator file
% data_path = './exports/';
% data_path = './exports_unitysigma/';
data_path = './export_fwd_IUSinterp/';
data_files = dir(fullfile(data_path,'*.mat'));
sim_type = "simulated";

dipole_N_ri_rec_distributions = [];

% for xi = 1:length(all_xbj_bins)
for xi = 5:5
    close all
    xbj_bin = string(all_xbj_bins(xi));

    for k = 1:numel(data_files)
        fname = data_files(k).name;
        if (contains(fname, xbj_bin) && contains(fname, data_type) && contains(fname, charm_opt) && contains(fname, r_steps_str))
            run_file = fname;
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
        % fnameb = strrep(fname,'-','_');
        fnameb = fname;
        if (contains(fnameb, sim_xbj) && contains(fnameb, fitname) && contains(fnameb, sim_type) && contains(fnameb, charm_opt) && contains(fnameb, r_steps_str))
            dip_file = fname;
            break
        end
    end
    if dip_file == ""
            ["failed to match dip_file!", fname, sim_xbj, fitname, sim_type, charm_opt, r_steps_str]
            return
    end
    % dip_file;
    dip_data = load(strcat(data_path, dip_file));
    ref_dip = dip_data.discrete_dipole_N;
    if (use_real_data)
        discrete_dipole_N = ref_dip;
    end
    %%
  
    ivec3= 1:r_steps;
    r_grid(end) = [];
    x = discrete_dipole_N;
    A = forward_op_A(:,ivec3);
    % bex = A*x';
    x = real_sigma*x(ivec3);
    bfit = A*x'; % bfit has numerical error from discretization
    % b is either the real data sigma_r, or one simulated by fit
    b_data = sigmar_vals'; % b is calculated by the C++ code, no error.
    q2vals = qsq_vals;
    [fitname, xbj_bin, r_steps,use_real_data,use_charm, length(b_data)]

    b_hera = [];
    b_errs = [];
    if use_real_data
        b_hera = sigmar_vals';
        b_errs = sigmar_errs'; % THIS IS NEEDED TO DO THE DATA \pm error reconstructions!
        % only do best reconst to b_err_upper and b_err_lower
    end
    
    % Simulating Gaussian 1% errors, and sampling datasets 
    
    % sample a dataset (b_gen_err_sampled) with 1% normal distrib error
    eta = 0.01;
    rng(80,"twister");
    % b_gen_err_sampled = b_data + eta.*b_data.*randn(length(b),1)
    % b_gen_err_sampled2 = b_data + eta.*b_data.*randn(length(b),1)
    % b_gen_err_sampled./b_gen_err_sampled2 % works! These are different.

    
    %%
    N=length(x);
    [L1,W1]=get_l(N,1);
    [L2,W2]=get_l(N,2);
    
    % classical 0th order tikhonov
    % [U,s,V] = csvd(A);
    % [UU,sm,XX] = csvd(A);
    
    % first order derivative operator
    [UU,sm,XX] = cgsvd(A,L1);
    
    % second order derivative operator
    % [UU2,sm2,XX2] = cgsvd(A,L2);
    % [UU,sm,XX] = cgsvd(A,L2);


    % principal reconstruction to actual data points
    X_tikh_principal = tikhonov(UU,sm,XX,b_data,lambda);
    errtik_p = zeros(size(lambda));
    for i = 1:length(lambda)
        errtik_p(i) = norm((b_data-A*X_tikh_principal(:,i)))/norm(b_data) + all_xbj_bins(xi)*1e-2*exp(-min(X_tikh_principal(:,i))); % add penalty for negative minimum
    end
    [mp,mIp]=min(errtik_p);
    rec_dip_principal = X_tikh_principal(:,mIp);
    sigmar_principal = A*rec_dip_principal;

    % bootstrapping reconstruction uncertainties
    array_over_dataset_samples_dipole_recs = [];
    array_over_dataset_samples_sigmar = [];
    % for j=1:10000
    parfor j=1:50000
        err = eta.*b_data.*randn(length(b_data),1);
        b = b_data + err;
    
        X_tikh = tikhonov(UU,sm,XX,b,lambda);
        errtik = zeros(size(lambda));
    
        if use_real_data
            X_tikh_upper = tikhonov(UU,sm,XX,b_err_upper,lambda);
            % errtik_upper = zeros(size(lambda));
            X_tikh_lower = tikhonov(UU,sm,XX,b_err_lower,lambda);
            % errtik_lower = zeros(size(lambda));
        end
        
        for i = 1:length(lambda)
            % errtik(i) = norm((x'-X_tikh(:,i)))/norm(x');
            % errtik(i) = norm((b-A*X_tikh(:,i)))/norm(b); % this is wrong, cannot really minimize against b since it has error
            if use_real_data
                % real data tests the calculated cross section sigma_r_rec
                % against the real data to constrain lambda
                errtik(i) = abs(norm((b-A*X_tikh(:,i))/(b_errs))-1); 
            else
                % simulated data refers agains the simulated dipole, which we
                % want to recover for the simulated sigma_r
                % errtik(i) = norm((x'-X_tikh(:,i)))/norm(x'); % compare
                % against the fit dipole x'
                % errtik(i) = norm((b-A*X_tikh(:,i)))/norm(b); % compare against simulated data. Relative error.
                errtik(i) = norm((b-A*X_tikh(:,i)))/norm(b) + all_xbj_bins(xi)*1e-2*exp(-min(X_tikh(:,i))); % add penalty for negative minimum
                % errtik(i) = norm((b-A*X_tikh(:,i)))/norm(b) + 1e-1*(1-sign(min(X_tikh(:,i)))); % This is perhaps too strong penalty?
                % errtik(i) = abs(norm((b-A*X_tikh(:,i))/(err))-1); 
            end
        end
    
        if use_real_data
            % [m,mI]=min(errtik);
            % flipped_errtik = flip(errtik)
            % for i=1:length(errtik)
            mI = find(errtik < 1, 1, "last") % lambda list is (should be) increasing, so we want the first lambda that hits chi^2/N= 1 taken from the large end
            m = errtik(mI)
        else
            [m,mI]=min(errtik);
            % mI = find(errtik < 1, 1, "last") % lambda list is (should be) increasing, so we want the first lambda that hits chi^2/N= 1 taken from the large end
            % m = errtik(mI)
        end
        
        rec_dip = X_tikh(:,mI);
        array_over_dataset_samples_dipole_recs(:,j) = rec_dip;
        array_over_dataset_samples_sigmar(:,j) = A*rec_dip;
    end % rec loop over dataset samples ends here
    
    dataset_sample_pdfs = [];
    dataset_sample_pdfs_sigmar = [];
    for j=1:length(x')
        rec_dips_at_rj = array_over_dataset_samples_dipole_recs(j,:)';
        sigmar_at_Qj = array_over_dataset_samples_dipole_recs(j,:)';
        % pd = fitdist(rec_dips_at_rj,"Normal");
        pd = fitdist(rec_dips_at_rj,'Kernel','Kernel','epanechnikov'); % see available methods with 'methods(pd)'
        pd_sig = fitdist(sigmar_at_Qj,'Kernel','Kernel','epanechnikov');
        dataset_sample_pdfs(j,:) = [mean(pd), std(pd)];
        dataset_sample_pdfs_sigmar(j,:) = [mean(pd_sig), std(pd_sig)];
    end

    % calculate sigma_r for each of the reconstructions to each data sample
    % calculate maximum difference to the principal data points (simu/real)
    % discard reconstructions whose discrepancy to the data points is
    % larger than the error of the data (there's no guarantee that any of
    % the reconstructions with random error hit close to the principal data
    % points after mapping A*rec_dip. Though, if the sampled error is small
    % and the sample set is close to the principal points on average, the
    % rec_dip should be close to the principal reconstruction? This would
    % provide a group of reconstructions, which should map back close to
    % the principal data points, within the assigned errors.(?)
    % 1. calc sigma_r = A*rec_dip_wth_err
    % 2. calc eps = max(sigmar - sigmar_data)
    % 3. if eps > data error at that point -> discard (perhaps conversely,
    % save valid reconstructions into a new array) (!)
    % 4. surviving reconstructions form the distribution of the
    % reconstructed dipole amplitude at each r_i. (calculated from
    % surviving dipole array)
    % SHOULDN'T ALL THE RECONSTRUCTIONS QUALITFY TO SURVIVE? They fit the
    % dataset with error closely -> eps < error by construction
    % It just doesn't make any sense to calculate the sigma_r uncertainty
    % from the upper and lower std bands, no single reconstruction goes to
    % either everywhere. (TEST THAT THEY DO?)
    
    % the result for the current xbj_bin is the full dataset_sample_pdfs,
    % which is the mean and std for each N(r_i)

    % dipole_N_ri_rec_distributions(xi,:,:) = dataset_sample_pdfs
    N_rec_ptw_mean = dataset_sample_pdfs(:,1);
    N_rec_std = dataset_sample_pdfs(:,2);
    N_rec_std_up = dataset_sample_pdfs(:,1) + dataset_sample_pdfs(:,2);
    N_rec_std_dn = dataset_sample_pdfs(:,1) - dataset_sample_pdfs(:,2);


    % plotting = false;
    plotting = true;
    if plotting
        figure(1) % mean reconstruction vs. ground truth
        % errorbar(r_grid', dataset_sample_pdfs(:,1), dataset_sample_pdfs(:,2))
        fill([r_grid';flipud(r_grid')], ...
             [dataset_sample_pdfs(:,1)-dataset_sample_pdfs(:,2);flipud(dataset_sample_pdfs(:,1)+dataset_sample_pdfs(:,2))], ...
             [.8 .9 .9],'linestyle','none')
        % plot(r_grid',x','-', ...
        %          r_grid',dataset_sample_pdfs(:,1),'--', ...
        %          r_grid',dataset_sample_pdfs(:,1)+dataset_sample_pdfs(:,2),'.', ...
        %          r_grid',dataset_sample_pdfs(:,1)-dataset_sample_pdfs(:,2),'.', ...
        %          'LineWidth',2)
        % semilogx(r_grid',x','-', ...
        loglog(r_grid',x','-', ...
                 r_grid',dataset_sample_pdfs(:,1),'--', ...
                 r_grid',dataset_sample_pdfs(:,1)+dataset_sample_pdfs(:,2),'.', ...
                 r_grid',dataset_sample_pdfs(:,1)-dataset_sample_pdfs(:,2),'.', ...
                 'LineWidth',2)
    end

    

    % delete(gcp("nocreate"));
    % return
    %%
    
    
    % EXPORTING RESULTS
    
    %% export reconstructed dipole statistics: pointwise distribution params
    %% and all relevant settings
    
    save2file = false;
    % save2file = true;
    if save2file
        if use_real_data
            reconst_type = "data_only";
        else
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
        % recon_path = "./reconstructions_IUSdip/";
        recon_path = "./reconstructions_gausserr/";
        f_exp_reconst = strjoin([recon_path 'recon_gausserr_' name '_xbj' xbj_bin '.mat'],"")
        % all_xbj_rec_pdf_means_std = dipole_N_ri_rec_distributions; %
        % maybe dont use this, keep saving to file one xbj at a time
        N_reconst = N_rec_ptw_mean;
        N_rec_std;
        N_rec_one_std_up = N_rec_std_up; % N_rec + std
        N_rec_one_std_dn = N_rec_std_dn; % N_rec - std
        N_fit = discrete_dipole_N;
        b_cpp_sim = b_data; % data generated in C++, no discretization error.
        b_fit = bfit; % = A*Nfit, this has discretization error.
        b_from_reconst = A*N_reconst; % prescription of the data by the reconstructred dipole.
        b_from_reconst_one_std_up = A*N_rec_one_std_up;
        b_from_reconst_one_std_dn = A*N_rec_one_std_dn;

    
        save(f_exp_reconst, ...
            "r_grid", "q2vals", ...
            "N_fit", "real_sigma",...
            "N_reconst", "N_rec_std", "N_rec_one_std_up", "N_rec_one_std_dn", ...
            "b_cpp_sim", "b_fit", "b_from_reconst", ...
            "b_hera", "b_errs", ...
            "b_from_reconst_one_std_up", "b_from_reconst_one_std_dn", ...
            "lambda", "lambda_type", ...
            "xbj_bin", "use_real_data", "use_charm", ...
            "run_file", "dip_file", ...
            "-nocompression","-v7")
    end
end
