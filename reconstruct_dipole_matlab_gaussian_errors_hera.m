% Include dependencies: AIRToolsII, Regtools
addpath(genpath('./dependencies'))
addpath(genpath("G:\My Drive\Postdoc MathPhys\Project 2 - Inverse dipole LO\HenriAnttiPaperv2"))

close all
clear all

parp = gcp;

all_xbj_bins = [1e-05, 0.0001, 0.00013, 0.0002, 0.00032, 0.0005, 0.0008, 0.001, 0.0013, 0.002, 0.0032, 0.005, 0.008, 0.01];
real_xbj_bins = [0.00013, 0.0002, 0.00032, 0.0005, 0.0008, 0.0013, 0.002, 0.0032, 0.005, 0.008, 0.013, 0.02, 0.032, 0.05, 0.08];
% TODO these bins probably change in different s bins, how to implement that? Discretized filename needs to include sqrt(s) as well?

% r_steps = 500;
r_steps = 256; % might not be quite good enough for high Q^2?
r_steps_str = strcat("r_steps",int2str(r_steps));

% forward operator data files
data_path = './export_hera_data/';
data_files = dir(fullfile(data_path,'*.mat'));

rec_methods = [
    "principal",
    "pkacz1",
    "pkacz2",
    "tikh0",
    "tikh2",
    "pcimmino1",
    "pcimmino2",
    ];

%%% real data settings
use_real_data = true;

data_type = "dis_inclusive";
data_name_key = "heraII_filtered";
if use_real_data
    all_xbj_bins = real_xbj_bins;
end

quark_mass_schemes = [
        "standard",
        "pole",
        "mqMpole",
        "mqmq",
        "mqMcharm",
        "mqMbottom",
        "mqMW",
    ];
% mscheme = quark_mass_schemes(1); % standard scheme for reference
% mscheme = quark_mass_schemes(5); % charm scale as the standard choice?
% mscheme = quark_mass_schemes(6); % n=10 prefers this over charm
mscheme = quark_mass_schemes(7); % W boson mass scale for high Q^2?

lambda_type = "SRN"; % strict+relaxed+noisy
lam1 = 1:0.5:9.5;
% lambda_noisy = [lam1*8e-4, lam1*1e-3]; % too noisy / complete over-fit breakdown?
% lambda_noisy = [lam1*3e-3]; % testing for noisy % very noisy at times, deprecate this, move towards relax
% lambda_relaxed = [lam1*8e-3, lam1*1e-2]; % OG relaxed, moving towards strict a little
% lambda_strict = [lam1*2e-2];
lambda_noisy = [lam1*5e-3];
lambda_relaxed = [lam1*9e-3, lam1*1e-2]; 
lambda_strict = [lam1*2e-2]; % this has been very good for the new safer strict, but n=13 has a hint of noise
% TODO a 'safe' option that works for all? relaxed closer to strict and strict to the 'safe' level?
% lambda_strict = [lam1*3e-2]; % this was better than 2e-2 for nn=10
% lambda_strict = [lam1*6e-2]; %
% lambda = [lam1*2e-2]; % fairly OK for HERA @xbj 0.08? looking for whats too stiff (TOO STIFF AT 0.00032)
% lambda = [lam1*5e-2]; % at 0.008 (close to the initial condition) Insanely strong preference for a big secondary peak!
lambda_t2 = [lam1*3e-2]; % This might be close for TIKH2 at 0.02??

% eps_neg_penalty=1e15; % this was working at low xi 1..3?
% eps_neg_penalty=1e-1; % This was working well for real data above xi=5, where suddenly big breakage
% eps_neg_penalty=1e-2; % strict default?
eps_neg_penalty=1e-4; % THIS IS NECESSARY FOR nn=5 !!! Even strict wasn't working with a higher penalty.




nn=5;
% nn=1;
% nn=13;
for xi = nn:nn
% for xi = 1:length(all_xbj_bins)
    close all
    xbj_bin = string(all_xbj_bins(xi));

    for k = 1:numel(data_files)
        fname = data_files(k).name;
        if (contains(fname, xbj_bin) && contains(fname, data_name_key) && contains(fname, mscheme) && contains(fname, r_steps_str))
            run_file = fname;
        end
    end
    load(strcat(data_path, run_file))
 
    rng(69,"twister");
    ivec3= 1:r_steps;
    r_grid(end) = [];
    A = forward_op_A(:,ivec3);
    b_data = sigmar_vals'; % b is calculated by the C++ code, no error.
    q2vals = qsq_vals;
    % TODO s-bin
    % todo data type inclusive, charm, diffractive etc
    [data_name_key, xbj_bin, r_steps, use_real_data, mscheme, length(b_data), length(q2vals), mean(sqrt(q2vals)), length(all_xbj_bins)]

    % real HERA data
    b_hera = sigmar_vals';
    b_errs = sigmar_errs';
    % TODO how to do correlated uncertainties??


    N=r_steps;
    [L1,W1]=get_l(N,1);
    [L2,W2]=get_l(N,2);

    % TODOs before starting to save results:
        % data types: inclusive vs charm (at least)
        % methods: implement save using different methods
        % s-bins: saving needs to denote which s-bin
    
   
    % first order derivative operator
    [UU,sm,XX] = cgsvd(A,L1);
    % second order derivative operator
    [UU2,sm2,XX2] = cgsvd(A,L2);


    % principal reconstruction to actual data points
    X_tikh_principal = tikhonov(UU,sm,XX,b_hera,lambda_strict);
    X_tikh_principal_relax = tikhonov(UU,sm,XX,b_hera,lambda_relaxed);
    X_tikh_principal_noisy = tikhonov(UU,sm,XX,b_hera,lambda_noisy);

    % runner up methods
    X_tikh2 = tikhonov(UU2,sm2,XX2,b_hera,lambda_t2);

    errtik_p = zeros(size(lambda_strict));
    errtik_prel = zeros(size(lambda_relaxed));
    errtik_pnoisy = zeros(size(lambda_noisy));
    errtik2 = zeros(size(lambda_t2));

    options.lbound = 0;
    % kmax = 5000; % Default? Hjørdis was using this.
    % kmax = 1000; % kacz2 worse than at 5k?
    % kmax1 = 200000;
    % kmax1 = 6000; % this has been VERY VERY good for kaczmarz for xi > 8 (0.0032) where it seems to overfit?
    kmax1 = 800;
    kmax2 = 5000;
    kmaxcim2 = kmax2;
    % xC = cimmino(A,b_hera,1:kmax,[],options);
    % xK = kaczmarz(A,b_hera,1:kmax,[],options);
    xC1 = PCimmino2(A,b_hera,kmax1,L1,W1);
    xK1 = PKaczmarz2(A,b_hera,kmax1,L1,W1);
    % xC2 = PCimmino2(A,b_hera,kmaxcim2,L2,W2);
    % xK2 = PKaczmarz2(A,b_hera,kmax2,L2,W2);

    
    for i = 1:length(lambda_strict)
        errtik_p(i) = norm((b_hera-A*X_tikh_principal(:,i)))/norm(b_hera) + eps_neg_penalty*(1-sign(min(X_tikh_principal(:,i))));
    end
    for i = 1:length(lambda_relaxed)
        errtik_prel(i) = norm((b_hera-A*X_tikh_principal_relax(:,i)))/norm(b_hera) + eps_neg_penalty*(1-sign(min(X_tikh_principal_relax(:,i))));
    end
    for i = 1:length(lambda_noisy)
        errtik_pnoisy(i) = norm((b_hera-A*X_tikh_principal_noisy(:,i)))/norm(b_hera) + eps_neg_penalty*(1-sign(min(X_tikh_principal_noisy(:,i))));
    end
    for i = i:length(lambda_t2)
        errtik2(i) = norm((b_hera-A*X_tikh2(:,i)))/norm(b_hera) + 10000*eps_neg_penalty*(1-sign(min(X_tikh2(:,i))));
    end
    [mp,mIp]=min(errtik_p);
    [mpr,mIpr]=min(errtik_prel);
    [mpn,mIpn]=min(errtik_pnoisy);
    rec_dip_principal_strict = X_tikh_principal(:,mIp);
    rec_dip_principal_relax = X_tikh_principal_relax(:,mIpr);
    rec_dip_principal_noisy = X_tikh_principal_noisy(:,mIpn);
    sigmar_principal_strict = A*rec_dip_principal_strict;
    sigmar_principal_relax = A*rec_dip_principal_relax;
    sigmar_principal_noisy = A*rec_dip_principal_noisy;

    % tikh comparison recs
    [m2,mI2]=min(errtik2);
    rec_tikh2 = X_tikh2(:,mI2);

    % comparison recs
    rec_cimmino1 = xC1(:,end);
    rec_kacz1 = xK1(:,end);
    % rec_kacz2 = xK2(:,end);

    % calculate sigmar for the comparison methods
    sigmar_tikh2 = A*rec_tikh2;
    sigmar_cimmino1 = A*rec_cimmino1;
    sigmar_kacz1 = A*rec_kacz1;
    % sigmar_kacz2 = A*rec_kacz2;


    
    % CALCULATE CHI^2 for the principal rec's agreement with the real data
    % (as one quantification of it's quality)
    % chi^2 = (teoria-mittaus)^2 / virhe^2 summattuna mittaus pisteiden yli
    % and also for top alternatives: Tikh2, Kacz1
    chisq_vec_strict = (sigmar_principal_strict - b_hera).^2 ./ b_errs.^2;
    chisq_vec_relax = (sigmar_principal_relax - b_hera).^2 ./ b_errs.^2;
    chisq_vec_noisy = (sigmar_principal_noisy - b_hera).^2 ./ b_errs.^2;
    chisq_vect2 = (sigmar_tikh2 - b_hera).^2 ./ b_errs.^2;
    chisq_veck1 = (sigmar_kacz1 - b_hera).^2 ./ b_errs.^2;
    chisq_cimm1 = (sigmar_cimmino1 - b_hera).^2 ./ b_errs.^2;
    % chisq = sum(chisq_vec);
    chisqt2 = sum(chisq_vect2);
    chisqk1 = sum(chisq_veck1);
    chisq_str_rel_noisy_vals = [sum(chisq_vec_strict), sum(chisq_vec_relax), sum(chisq_vec_noisy)];
    chisq_over_N_strict = sum(chisq_vec_strict) / length(chisq_vec_strict);
    chisq_over_N_relax = sum(chisq_vec_relax) / length(chisq_vec_relax);
    chisq_over_N_noisy = sum(chisq_vec_noisy) / length(chisq_vec_noisy);
    chisq_over_Nt2 = chisqt2 / length(chisq_vect2);
    chisq_over_Nk1 = chisqk1 / length(chisq_veck1);
    chisq_over_Ncim1 = sum(chisq_cimm1) / length(chisq_cimm1);
    [chisq_over_N_strict, chisq_over_N_relax, chisq_over_N_noisy, chisq_over_Nt2, chisq_over_Nk1, chisq_over_Ncim1]



    %%%%%%%
    %%% BOOTSTRAPPING UNCERTAINTIES
    %%%%%%%

    % bootstrapping reconstruction uncertainties with Uncorrelated Experiment Uncertainties.
    array_over_dataset_samples_dipole_recs = [];
    array_over_dataset_samples_sigmar = [];
    NUM_SAMPLES = 1000;
    parfor j=1:NUM_SAMPLES
        % TODO need to sample error point specific distributions! -> redo err and b.
        % err = eta.*b_data.*randn(length(b_data),1); %%% relative error for simulated data (paper 1)
        err = b_errs.*randn(length(b_hera),1);
        b = b_hera + err;
    
        X_tikh = tikhonov(UU,sm,XX,b,lambda_strict);
        errtik = zeros(size(lambda_strict));
    
        
        for i = 1:length(lambda_strict)
            % errtik(i) = norm((x'-X_tikh(:,i)))/norm(x');
            % errtik(i) = norm((b-A*X_tikh(:,i)))/norm(b); % this is wrong, cannot really minimize against b since it has error
            if use_real_data
                % real data tests the calculated cross section sigma_r_rec
                % against the real data to constrain lambda
                % errtik(i) = abs(norm((b-A*X_tikh(:,i))/(b_errs))-1);
                errtik(i) = norm((b-A*X_tikh(:,i)))/norm(b) + eps_neg_penalty*(1-sign(min(X_tikh(:,i)))); % This is perhaps too strong penalty?
            else
                % simulated data refers agains the simulated dipole, which we
                % want to recover for the simulated sigma_r
                % errtik(i) = norm((x'-X_tikh(:,i)))/norm(x'); % compare
                % against the fit dipole x'
                % errtik(i) = norm((b-A*X_tikh(:,i)))/norm(b); % compare against simulated data. Relative error.
                % errtik(i) = norm((b-A*X_tikh(:,i)))/norm(b) + all_xbj_bins(xi)*1e-2*exp(-min(X_tikh(:,i))); % add penalty for negative minimum
                errtik(i) = norm((b-A*X_tikh(:,i)))/norm(b) + eps_neg_penalty*(1-sign(min(X_tikh(:,i)))); % This is perhaps too strong penalty?
                % errtik(i) = abs(norm((b-A*X_tikh(:,i))/(err))-1); 
            end
        end
    
        if use_real_data
            [m,mI]=min(errtik); % todo compare with the "last" method of choosing the lambda?
            % mI = find(errtik < 1, 1, "last") % lambda list is (should be) increasing, so we want the first lambda that hits chi^2/N= 1 taken from the large end
            % m = errtik(mI)
        else
            [m,mI]=min(errtik);
        end
        
        rec_dip = X_tikh(:,mI);
        array_over_dataset_samples_dipole_recs(:,j) = rec_dip;
        array_over_dataset_samples_sigmar(:,j) = A*rec_dip;
    end % rec loop over dataset samples ends here
    
    % Reconstruction statistics / bootstrapping for pointwise distributions
    dataset_sample_pdfs = [];
    dataset_sample_pdfs_sigmar = [];
    p_tails_95 = [0.025, 0.975];
    p_tails_682 = [0.159, 0.841];
    for j=1:r_steps
        rec_dips_at_rj = array_over_dataset_samples_dipole_recs(j,:)';
        dip_rm_outliers = rmoutliers(rec_dips_at_rj, "percentiles", [1 99]);
        pd = fitdist(dip_rm_outliers,'Kernel','Kernel','epanechnikov'); % see available methods with 'methods(pd)'
        dip_icdf_vals_95 = icdf(pd, p_tails_95);
        if abs(dip_icdf_vals_95(1)) > 200
            dip_icdf_vals_95(1) = nan;
        end
        if abs(dip_icdf_vals_95(2)) > 200
            dip_icdf_vals_95(2) = nan;
        end
        dip_icdf_vals_682 = icdf(pd, p_tails_682);
        if abs(dip_icdf_vals_682(1)) > 200
            dip_icdf_vals_682(1) = nan;
        end
        if abs(dip_icdf_vals_682(2)) > 200
            dip_icdf_vals_682(2) = nan;
        end
        dataset_sample_pdfs(j,:) = [mean(pd), dip_icdf_vals_682(1), dip_icdf_vals_682(2), dip_icdf_vals_95(1), dip_icdf_vals_95(2)];
    end
    for j=1:length(q2vals)
        sigmar_at_Qj = array_over_dataset_samples_sigmar(j,:)';
        % sig_rm_outliers = rmoutliers(sigmar_at_Qj);
        pd_sig = fitdist(sigmar_at_Qj,'Kernel','Kernel','epanechnikov');
        sig_icdf_vals_95 = icdf(pd_sig, p_tails_95);
        sig_icdf_vals_682 = icdf(pd_sig, p_tails_682);
        dataset_sample_pdfs_sigmar(j,:) = [mean(pd_sig), sig_icdf_vals_682(1), sig_icdf_vals_682(2), sig_icdf_vals_95(1), sig_icdf_vals_95(2)];
    end
    
    N_rec_principal = rec_dip_principal_strict;
    N_rec_ptw_mean = dataset_sample_pdfs(:,1);
    N_rec_CI682_up = dataset_sample_pdfs(:,3); % 68.2% confidence interval upper limit
    N_rec_CI682_dn = dataset_sample_pdfs(:,2); % 68.2% c.i. lower limit
    N_rec_CI95_up = dataset_sample_pdfs(:,5); % 95% confidence interval upper limit
    N_rec_CI95_dn = dataset_sample_pdfs(:,4); % 95% c.i. lower limit
    % if any(N_rec_ptw_mean <= 0) % TEST IF THE MEAN RECONSTRUCTION IS POSITIVE
    %     ["NON-POSITIVE PRINCIPAL reconstruction at", xbj_bin, r_grid(1), rec_dip_principal(1), N_rec_ptw_mean(1)]
    %     % return
    % end
    % ["Rec accuracy", r_grid(10), N_rec_principal(10)/x(10), r_grid(20), N_rec_principal(20)/x(20), r_grid(50), N_rec_principal(50)/x(50), r_grid(60), N_rec_principal(60)/x(60), r_grid(100), N_rec_principal(100)/x(100), r_grid(200), N_rec_principal(200)/x(200)]

    sigmar_principal_strict;
    sigmar_ptw_mean = A*N_rec_ptw_mean;
    sigmar_mean = dataset_sample_pdfs_sigmar(:,1);
    sigmar_CI682_up = dataset_sample_pdfs_sigmar(:,3);
    sigmar_CI682_dn = dataset_sample_pdfs_sigmar(:,2);
    sigmar_CI95_up = dataset_sample_pdfs_sigmar(:,5);
    sigmar_CI95_dn = dataset_sample_pdfs_sigmar(:,4);

    % plotting = false;
    plotting = true;
    if plotting
        figure(1) % rec_princip vs. mean reconstruction vs. ground truth
        % errorbar(r_grid', dataset_sample_pdfs(:,1), dataset_sample_pdfs(:,2))
        % fill([r_grid';flipud(r_grid')], ...
        %      [N_rec_std_dn;flipud(N_rec_std_up)], ...
        %      [.8 .9 .9],'linestyle','none')
        % semilogx(r_grid',x','-', ...
        % plot(r_grid',x','-', ...
        semilogx(r_grid',N_rec_principal,'--', "DisplayName", "principal", "Color","blue")
        hold on
        loglog(r_grid',abs(N_rec_principal),':', "DisplayName", "ABS principal", "Color","blue")
        loglog(r_grid',N_rec_ptw_mean,'-.', "DisplayName", "mean", "Color","white")
        loglog(r_grid',rec_dip_principal_relax,'-.', "DisplayName", "relax", "Color", "#FFA500")
        loglog(r_grid',rec_dip_principal_noisy,'-.', "DisplayName", "noisy", "Color","#FF69B4")
        loglog(r_grid',N_rec_CI95_up,':', "DisplayName", "95 up", "Color","Red")
        loglog(r_grid',N_rec_CI95_dn,':', "DisplayName", "95 dn", "Color","Red")
        loglog(r_grid',N_rec_CI682_up,':', "DisplayName", "68 up", "Color","Green")
        loglog(r_grid',N_rec_CI682_dn,':', "DisplayName", "68 dn", "Color","Green")
        % loglog(r_grid',rec_tikh0,'-.', "DisplayName", "tikh0")
        % loglog(r_grid',rec_tikh2,'-.', "DisplayName", "tikh2", "Color","magenta")
        % loglog(r_grid',abs(rec_tikh2),'-.', "DisplayName", "abs tikh2", "Color","cyan")
        % % loglog(r_grid',rec_cimmino,'--', "DisplayName", "cimmino")
        loglog(r_grid',rec_cimmino1,'-.', "DisplayName", "cimmino1")
        % % loglog(r_grid',rec_cimmino2,'--', "DisplayName", "cimmino2")
        % % loglog(r_grid',rec_kacz,'--', "DisplayName", "kacz")
        loglog(r_grid',rec_kacz1,'-.', "DisplayName", "kacz1", "Color","Yellow")
        % % loglog(r_grid',rec_kacz2,'--', "DisplayName", "kacz2", "Color","white")
        hold off
               

        figure(2)
        % [size(q2vals'), size(b_data), size(sigmar_principal), size(sigmar_ptw_mean), size(sigmar_CI_up), size(sigmar_CI_dn),] 
        semilogx(q2vals',b_hera,'-', 'DisplayName',"hera")
        hold on
        errorbar(q2vals', b_hera, b_errs, '')
        % hold on
        semilogx(q2vals',sigmar_principal_strict,'-.', 'DisplayName',"principal_strict")
        semilogx(q2vals',sigmar_principal_relax,'-.', 'DisplayName',"principal_relax")
        semilogx(q2vals',sigmar_principal_noisy,'-.', 'DisplayName',"principal_noisy")
        semilogx(q2vals',sigmar_ptw_mean,':', 'DisplayName',"mean")
        semilogx(q2vals',sigmar_CI95_up,':', "DisplayName", "95 up", "Color","Red")
        semilogx(q2vals',sigmar_CI95_dn,':', "DisplayName", "95 dn", "Color","Red")
        semilogx(q2vals',sigmar_CI682_up,':', "DisplayName", "68 up", "Color","Green")
        semilogx(q2vals',sigmar_CI682_dn,':', "DisplayName", "68 dn", "Color","Green")
        % semilogx(q2vals',sigmar_kacz,'--', "DisplayName", "skacz")
        semilogx(q2vals',sigmar_tikh2,'--', "DisplayName", "tikh2")
        semilogx(q2vals',sigmar_cimmino1,'--', "DisplayName", "cimm1")
        % semilogx(q2vals',sigmar_cimmino2,'--', "DisplayName", "cimm2")
        semilogx(q2vals',sigmar_kacz1,'--', "DisplayName", "skacz1")
        % semilogx(q2vals',sigmar_kacz2,'--', "DisplayName", "skacz2")
        hold off
    end

    
    % EXPORTING RESULTS
    
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
        flavor_string = mscheme;
        name = [data_name, '_', reconst_type, '_', flavor_string, '_', lambda_type];
        % recon_path = "./reconstructions_IUSdip/";
        recon_path = "./reconstructions_gausserr/";
        f_exp_reconst = strjoin([recon_path 'hera_recon_gausserr_' r_steps '_' name '_xbj' xbj_bin '.mat'],"")
        N_reconst = N_rec_principal;
        N_rec_one_std_up = N_rec_CI682_up; % N_rec + std
        N_rec_one_std_dn = N_rec_CI682_dn; % N_rec - std
        % N_fit = discrete_dipole_N;
        chisq_str_rel_noisy_vals;
        % b_cpp_sim = b_data; % data generated in C++, no discretization error.
        % b_fit = bfit; % = A*Nfit, this has discretization error.
        b_from_reconst = sigmar_principal; % prescription of the data by the reconstructred dipole.

        save(f_exp_reconst, ...
            "r_grid", "r_steps", "q2vals", ...
            "N_reconst", "N_rec_one_std_up", "N_rec_one_std_dn", ...
            "b_from_reconst", ...
            "b_hera", "b_errs", ...
            "N_rec_principal", ...
            "N_rec_ptw_mean", ...
            "N_rec_CI682_up", "N_rec_CI682_dn", ...
            "N_rec_CI95_up", "N_rec_CI95_dn", ...
            "sigmar_principal", ...
            "sigmar_ptw_mean", ...
            "sigmar_mean", ...
            "sigmar_CI682_up", "sigmar_CI682_dn", ...
            "sigmar_CI95_up", "sigmar_CI95_dn", ...
            "lambda", "lambda_type", "NUM_SAMPLES", "eps_neg_penalty", ...
            "xbj_bin", "use_real_data", "mscheme", ...
            "run_file", ...
            "-nocompression","-v7")
    end
end
