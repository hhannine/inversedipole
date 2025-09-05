% Include dependencies: AIRToolsII, Regtools
addpath(genpath('./dependencies'))
addpath(genpath("C:\Users\hana_\My Drive\Postdoc MathPhys\Project 2 - Inverse dipole LO\HenriAnttiPaperv2"))

close all
clear all

parp = gcp;

s_bins = [318.1, 300.3, 251.5]; % 224.9 bin had no viable xbj bins at all
s_bin = s_bins(1);
s_str = "s" + string(s_bin);

all_xbj_bins = [1e-05, 0.0001, 0.00013, 0.0002, 0.00032, 0.0005, 0.0008, 0.001, 0.0013, 0.002, 0.0032, 0.005, 0.008, 0.01];
real_xbj_bins = [0.00013, 0.0002, 0.00032, 0.0005, 0.0008, 0.0013, 0.002, 0.0032, 0.005, 0.008, 0.013, 0.02, 0.032, 0.05, 0.08];
% TODO these bins probably change in different s bins, how to implement that? Discretized filename needs to include sqrt(s) as well?
s300_xbj_bins = [0.0002, 0.02];
s251_xbj_bins = [0.032, 0.05, 0.08, 0.13, 0.18, 0.4];

if s_bin == s_bins(2)
    real_xbj_bins = s300_xbj_bins;
elseif s_bin == s_bins(3)
    real_xbj_bins = s251_xbj_bins;
end

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

data_type = "dis_inclusive"; % vs. dis_charm, dis_bottom, diff_dis_inclusive
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
mscheme = quark_mass_schemes(5); % charm scale as the standard choice?
% mscheme = quark_mass_schemes(6); % n=10 prefers this over charm
% mscheme = quark_mass_schemes(7); % W boson mass scale for high Q^2?

lambda_type = "lambdaSRN"; % strict+relaxed+noisy
lam1 = 1:0.1:9.9;
% lambda_noisy = [lam1*8e-4, lam1*1e-3]; % too noisy / complete over-fit breakdown?
% lambda_noisy = [lam1*3e-3]; % testing for noisy % very noisy at times, deprecate this, move towards relax
% lambda_relaxed = [lam1*8e-3, lam1*1e-2]; % OG relaxed, moving towards strict a little
% lambda_strict = [lam1*2e-2];
lambda_noisy = [lam1*5e-3];
lambda_relaxed = [lam1*5e-3, lam1*1e-2, lam1*1e-1]; 
lambda_strict = lambda_relaxed;
% lambda_strict = [lam1*1e-2]; % smaller x?
% lambda_strict = [lam1*1e-2]; % 2e-2 is the default strict % this has been very good for the new safer strict, but n=13 has a hint of noise
% lambda_strict = [lam1*2e-2, lam1*1e-1]; % for large x?
% lambda_strict = [lam1*3e-2]; % for large x?
% TODO a 'safe' option that works for all? relaxed closer to strict and strict to the 'safe' level?
% lambda_strict = [lam1*3e-2]; % this was better than 2e-2 for nn=10
% lambda_strict = [lam1*100e-2]; % the alternate s_bins require a VERY large lambda due to low amount of points
% lambda = [lam1*2e-2]; % fairly OK for HERA @xbj 0.08? looking for whats too stiff (TOO STIFF AT 0.00032)
% lambda = [lam1*5e-2]; % at 0.008 (close to the initial condition) Insanely strong preference for a big secondary peak!
lambda_t2 = [lam1*3e-2]; % This might be close for TIKH2 at 0.02??

% eps_neg_penalty=1e15; % this was working at low xi 1..3?
% eps_neg_penalty=1e-1; % This was working well for real data above xi=5, where suddenly big breakage
% eps_neg_penalty=1e-2; % strict default?
eps_neg_penalty=1e-4; % THIS IS NECESSARY FOR nn=5 !!! Even strict wasn't working with a higher penalty.

%           1  2  3   4  5   6  7  8   9  10  11  12  13  14  15
q_cuts = [1.5, 4, 5, 9, 12, 15, 22, 22, 22, 22, 22, 20, 30, 90, 100]; % bins selected to cut small-r peak

save2file = false;
save2file = true;
if save2file == true
    n0 = 1;
    nend = length(all_xbj_bins);
    plotting = false;
else
    % nn=15; % probably too weird to use? 14 perhaps even worse?
    % nn=3;
    nn=10; % big sub-peak
    % nn=8; % smaller but definite sub-peak, needed to drop lambda_strict to 1e-2 to see it.
    % nn=5;
    n0 = nn;
    nend = nn;
    plotting = true;
end

% for xi = 1:length(all_xbj_bins)
for xi = n0:nend
    close all
    xbj_bin = string(all_xbj_bins(xi));

    for k = 1:numel(data_files)
        fname = data_files(k).name;
        if (contains(fname, xbj_bin) && contains(fname, data_name_key) && contains(fname, mscheme) && contains(fname, r_steps_str) && contains(fname, s_str))
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
    % real HERA data
    b_hera = sigmar_vals';
    b_errs = sigmar_errs';

    % testing Q binning -- Careful, limiting upper limit reduces the number of points, and the reconstrctuion can break with too few points!
    q_cut = 0; % no cut low or high
    use_low_Q_cut = false;
    % use_low_Q_cut = true;
    if use_low_Q_cut
        if q2vals(1) < q_cuts(xi)
            "Smallest Q^2 point larger than cut!"
        end
        q2vals
        % lambda_strict = [lam1*2e-2]; % for 2? -> many different bins, probably should do the 'find largest that's good enough' and take lambda as a nuisance parameter?
        % lambda_strict = [lam1*3e-2]; % for 3?
        % lambda_strict = [lam1*4e-2]; % for 5! 4?
        % lambda_strict = [lam1*5e-2]; % for 7, 8?
        % lambda_strict = [lam1*8e-2]; % for n=13, 14 (very tiny sigma0)
        % q_cut = 15; % q^2 = 5 seems like a good cut (for n=10), n=9 peak starts to go at cut=15
        % q_cut = 9; % n=5, small x in general, needs lower cut (cut=5 still has peak for n=5, cut=9 is good: peak goes, negativity goes!)
        q_cut = q_cuts(xi);
        qn = find(q2vals>q_cut,1,'first');
        q2vals = q2vals(qn:length(q2vals)) % sub-bump disappears with a cut at 18 GeV?
        A = A(qn:length(q2vals)+qn-1,:);
        b_hera = b_hera(qn:length(q2vals)+qn-1);
        b_errs = b_errs(qn:length(q2vals)+qn-1);
    end
    use_high_Q_cut = false;
    % use_high_Q_cut = true;
    q_high=22;
    if use_high_Q_cut
        if q2vals(end) > q_high
            "highest Q^2 point lower than cut!"
        end
        eps_neg_penalty=1e10;
        % lambda_noisy = [lam1*3e-3];
        q2vals
        % lambda_strict = [lam1*2e-2]; % for 2? -> many different bins, probably should do the 'find largest that's good enough' and take lambda as a nuisance parameter?
        % lambda_strict = [lam1*3e-2]; % for 3?
        % lambda_strict = [lam1*4e-2]; % for 5! 4?
        % lambda_strict = [lam1*6e-2]; % for 7, 8?
        % lambda_strict = [lam1*8e-2]; % for n=13, 14 (very tiny sigma0)
        % q_cut = 15; % q^2 = 5 seems like a good cut (for n=10), n=9 peak starts to go at cut=15
        % q_cut = 9; % n=5, small x in general, needs lower cut (cut=5 still has peak for n=5, cut=9 is good: peak goes, negativity goes!)
        q_cut = q_high;
        qn = find(q2vals<q_cut,1,'last');
        q2vals = q2vals(1:qn)
        A = A(1:qn,:);
        b_hera = b_hera(1:qn);
        b_errs = b_errs(1:qn);
    end
    [xi, data_name_key, s_bin, xbj_bin, r_steps, use_real_data, mscheme, length(q2vals), min(q2vals), mean(sqrt(q2vals)), mean(q2vals), length(all_xbj_bins)]


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
        % errtik_p(i) = norm((b_hera-A*X_tikh_principal(:,i)))/norm(b_hera) + eps_neg_penalty*(1-sign(min(X_tikh_principal(:,i))));
        chisq_v = (A*X_tikh_principal(:,i) - b_hera).^2 ./ b_errs.^2;
        errtik_p(i) = abs(sum(chisq_v) / (length(chisq_v)-1) - 1.00);
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
    % alternate selection criterion which looks for the "stiffest" lambda that works well enough:
    % mIp = find(errtik_p > 1, 1, "first") % lambda list is (should be) increasing, so we want the first lambda that hits chi^2/N= 1 taken from the large end
    % mp = errtik_p(mIp)
    lambda_unity_chisq = lambda_strict(mIp)
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
    chisq_over_N_strict = sum(chisq_vec_strict) / (length(chisq_vec_strict)-1);
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
    array_over_dataset_samples_dipole_recs_rel = [];
    array_over_dataset_samples_sigmar_rel = [];
    NUM_SAMPLES = 1000;
    parfor j=1:NUM_SAMPLES
        % TODO need to sample error point specific distributions! -> redo err and b.
        % err = eta.*b_data.*randn(length(b_data),1); %%% relative error for simulated data (paper 1)
        err = b_errs.*randn(length(b_hera),1);
        b = b_hera + err;
    
        X_tikh = tikhonov(UU,sm,XX,b,lambda_strict);
        errtik = zeros(size(lambda_strict));
        X_tikh_rel = tikhonov(UU,sm,XX,b,lambda_relaxed);
        errtik_rel = zeros(size(lambda_relaxed));
    
        
        for i = 1:length(lambda_strict)
            % errtik(i) = abs(norm((b-A*X_tikh(:,i))/(b_errs))-1);
            % errtik(i) = norm((b-A*X_tikh(:,i)))/norm(b) + eps_neg_penalty*(1-sign(min(X_tikh(:,i))));
            chisq_v = (A*X_tikh_principal(:,i) - b_hera).^2 ./ b_errs.^2;
            errtik(i) = abs(sum(chisq_v) / (length(chisq_v)-1) - 1.00);
        end
        for i = 1:length(lambda_relaxed)
            % errtik(i) = abs(norm((b-A*X_tikh(:,i))/(b_errs))-1);
            errtik_rel(i) = norm((b-A*X_tikh_rel(:,i)))/norm(b) + eps_neg_penalty*(1-sign(min(X_tikh_rel(:,i))));
        end
    
        [m,mI]=min(errtik);
        [m_rel,mI_rel]=min(errtik_rel);
        % alternate selection criterion which looks for the "stiffest" lambda that works well enough:
        % mI = find(errtik < 1, 1, "last") % lambda list is (should be) increasing, so we want the first lambda that hits chi^2/N= 1 taken from the large end
        % m = errtik(mI)

        rec_dip = X_tikh(:,mI);
        array_over_dataset_samples_dipole_recs(:,j) = rec_dip;
        array_over_dataset_samples_sigmar(:,j) = A*rec_dip;
        rec_dip_rel = X_tikh_rel(:,mI_rel);
        array_over_dataset_samples_dipole_recs_rel(:,j) = rec_dip_rel;
        array_over_dataset_samples_sigmar_rel(:,j) = A*rec_dip_rel;
    end % rec loop over dataset samples ends here
    
    % Reconstruction statistics / bootstrapping for pointwise distributions
    % dataset_sample_pdfs = [];
    % dataset_sample_pdfs_sigmar = [];
    % dataset_sample_pdfs_rel = [];
    % dataset_sample_pdfs_sigmar_rel = [];
    dataset_sample_pdfs = zeros([length(r_steps),5]);
    dataset_sample_pdfs_sigmar = zeros([length(r_steps),5]);
    dataset_sample_pdfs_rel = zeros([length(q2vals),5]);
    dataset_sample_pdfs_sigmar_rel = zeros([length(q2vals),5]);
    p_tails_95 = [0.025, 0.975];
    p_tails_682 = [0.159, 0.841];
    for j=1:r_steps
        rec_dips_at_rj = array_over_dataset_samples_dipole_recs(j,:)';
        dip_rm_outliers = rmoutliers(rec_dips_at_rj, "percentiles", [1.5 98.5]);
        pd = fitdist(dip_rm_outliers,'Kernel','Kernel','epanechnikov'); % see available methods with 'methods(pd)'
        dip_icdf_vals_95 = icdf(pd, p_tails_95);
        rec_dips_at_rj_rel = array_over_dataset_samples_dipole_recs_rel(j,:)';
        dip_rm_outliers_rel = rmoutliers(rec_dips_at_rj_rel, "percentiles", [1.5 98.5]);
        pd_rel = fitdist(dip_rm_outliers_rel,'Kernel','Kernel','epanechnikov'); % see available methods with 'methods(pd)'
        dip_icdf_vals_95_rel = icdf(pd_rel, p_tails_95);
        dip_icdf_vals_682 = icdf(pd, p_tails_682);
        dip_icdf_vals_682_rel = icdf(pd_rel, p_tails_682);
        % if abs(dip_icdf_vals_95(1)) > 200
        %     dip_icdf_vals_95(1) = nan;
        % end
        % if abs(dip_icdf_vals_95(2)) > 200
        %     dip_icdf_vals_95(2) = nan;
        % end
        % if abs(dip_icdf_vals_682(1)) > 200
        %     dip_icdf_vals_682(1) = nan;
        % end
        % if abs(dip_icdf_vals_682(2)) > 200
        %     dip_icdf_vals_682(2) = nan;
        % end
        dataset_sample_pdfs(j,:) = [mean(pd), dip_icdf_vals_682(1), dip_icdf_vals_682(2), dip_icdf_vals_95(1), dip_icdf_vals_95(2)];
        dataset_sample_pdfs_rel(j,:) = [mean(pd_rel), dip_icdf_vals_682_rel(1), dip_icdf_vals_682_rel(2), dip_icdf_vals_95_rel(1), dip_icdf_vals_95_rel(2)];
    end
    for j=1:length(q2vals)
        sigmar_at_Qj = array_over_dataset_samples_sigmar(j,:)';
        pd_sig = fitdist(sigmar_at_Qj,'Kernel','Kernel','epanechnikov');
        sig_icdf_vals_95 = icdf(pd_sig, p_tails_95);
        sig_icdf_vals_682 = icdf(pd_sig, p_tails_682);
        dataset_sample_pdfs_sigmar(j,:) = [mean(pd_sig), sig_icdf_vals_682(1), sig_icdf_vals_682(2), sig_icdf_vals_95(1), sig_icdf_vals_95(2)];
        sigmar_at_Qj_rel = array_over_dataset_samples_sigmar_rel(j,:)';
        pd_sig_rel = fitdist(sigmar_at_Qj_rel,'Kernel','Kernel','epanechnikov');
        sig_icdf_vals_95_rel = icdf(pd_sig_rel, p_tails_95);
        sig_icdf_vals_682_rel = icdf(pd_sig_rel, p_tails_682);
        dataset_sample_pdfs_sigmar_rel(j,:) = [mean(pd_sig_rel), sig_icdf_vals_682_rel(1), sig_icdf_vals_682_rel(2), sig_icdf_vals_95_rel(1), sig_icdf_vals_95_rel(2)];
    end
    
    N_rec_principal = rec_dip_principal_strict;
    N_rec_ptw_mean = dataset_sample_pdfs(:,1);
    N_rec_CI682_up = dataset_sample_pdfs(:,3); % 68.2% confidence interval upper limit
    N_rec_CI682_dn = dataset_sample_pdfs(:,2); % 68.2% c.i. lower limit
    N_rec_CI95_up = dataset_sample_pdfs(:,5); % 95% confidence interval upper limit
    N_rec_CI95_dn = dataset_sample_pdfs(:,4); % 95% c.i. lower limit
    N_rec_principal_relax = rec_dip_principal_relax;
    N_rec_ptw_mean_relax = dataset_sample_pdfs_rel(:,1);
    N_rec_CI682_up_relax = dataset_sample_pdfs_rel(:,3); % 68.2% confidence interval upper limit
    N_rec_CI682_dn_relax = dataset_sample_pdfs_rel(:,2); % 68.2% c.i. lower limit
    N_rec_CI95_up_relax = dataset_sample_pdfs_rel(:,5); % 95% confidence interval upper limit
    N_rec_CI95_dn_relax = dataset_sample_pdfs_rel(:,4); % 95% c.i. lower limit
    N_rec_principal_noisy = rec_dip_principal_noisy;
    [max(N_rec_principal), max(N_rec_principal_relax), max(N_rec_principal_noisy)]
    if min(N_rec_principal) < 0
        [M,I] = min(N_rec_principal);
        ["Negative principal", M, M/max(N_rec_principal), r_grid(I)]
    end
    [N_max_strict, max_i] = max(rec_dip_principal_strict);
    [N_max_relax, max_i_rel] = max(rec_dip_principal_relax);
    [N_max_noisy, max_i_noisy] = max(rec_dip_principal_noisy);
    r_Nmax_strict = r_grid(max_i);
    r_Nmax_rel = r_grid(max_i_rel);
    r_Nmax_noisy = r_grid(max_i_noisy);
    N_max_strict_ci = [N_rec_CI682_dn(max_i), N_rec_CI682_up(max_i), N_rec_CI95_dn(max_i), N_rec_CI95_up(max_i)];
    N_max_relax_ci = [N_rec_CI682_dn_relax(max_i_rel), N_rec_CI682_up_relax(max_i_rel), N_rec_CI95_dn_relax(max_i_rel), N_rec_CI95_up_relax(max_i_rel)];
    N_max_data_strict = [N_max_strict, r_Nmax_strict, N_max_strict_ci];
    N_max_data_relax = [N_max_relax, r_Nmax_rel, N_max_relax_ci];

    sigmar_principal_strict;
    sigmar_ptw_mean = A*N_rec_ptw_mean;
    sigmar_mean = dataset_sample_pdfs_sigmar(:,1);
    sigmar_CI682_up = dataset_sample_pdfs_sigmar(:,3);
    sigmar_CI682_dn = dataset_sample_pdfs_sigmar(:,2);
    sigmar_CI95_up = dataset_sample_pdfs_sigmar(:,5);
    sigmar_CI95_dn = dataset_sample_pdfs_sigmar(:,4);
    sigmar_principal_relax;
    sigmar_ptw_mean_relax = A*N_rec_ptw_mean_relax;
    sigmar_mean_relax = dataset_sample_pdfs_sigmar_rel(:,1);
    sigmar_CI682_up_relax = dataset_sample_pdfs_sigmar_rel(:,3);
    sigmar_CI682_dn_relax = dataset_sample_pdfs_sigmar_rel(:,2);
    sigmar_CI95_up_relax = dataset_sample_pdfs_sigmar_rel(:,5);
    sigmar_CI95_dn_relax = dataset_sample_pdfs_sigmar_rel(:,4);
    sigmar_principal_noisy;

    % plotting = false;
    % plotting = true;
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
        loglog(r_grid',N_rec_CI95_up_relax,'.', "DisplayName", "95 up relax", "Color","Red")
        loglog(r_grid',N_rec_CI95_dn_relax,'.', "DisplayName", "95 dn rel", "Color","Red")
        loglog(r_grid',N_rec_CI682_up_relax,'.', "DisplayName", "68 up rel", "Color","Green")
        loglog(r_grid',N_rec_CI682_dn_relax,'.', "DisplayName", "68 dn rel", "Color","Green")
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
            data_name = "heraII_"+data_type+s_str;
        else
            data_name = "sim";
        end
        q_cut_str = "no_Q_cut_";
        if use_low_Q_cut == true
            q_cut_str = "cut_low_Q_";
        elseif use_high_Q_cut == true
            q_cut_str = "cut_high_Q_";
        end
        flavor_string = mscheme;
        name = [data_name, '_', reconst_type, '_', flavor_string, '_', lambda_type, '_', q_cut_str];
        recon_path = "./reconstructions_hera_uq/";
        f_exp_reconst = strjoin([recon_path 'hera_recon_uq_' r_steps '_' name '_xbj' xbj_bin '.mat'],"")
        N_reconst = N_rec_principal;
        N_rec_one_std_up = N_rec_CI682_up; % N_rec + std
        N_rec_one_std_dn = N_rec_CI682_dn; % N_rec - std
        chisq_str_rel_noisy_vals;
        b_from_reconst = sigmar_principal_strict; % prescription of the data by the reconstructred dipole.

        save(f_exp_reconst, ...
            "r_grid", "r_steps", "q2vals", ...
            "N_reconst", "N_rec_one_std_up", "N_rec_one_std_dn", ...
            "b_from_reconst", "b_hera", "b_errs", ...
            "N_rec_principal", "N_rec_ptw_mean", "N_rec_CI682_up", "N_rec_CI682_dn", "N_rec_CI95_up", "N_rec_CI95_dn", ...
            "N_rec_principal_relax", "N_rec_ptw_mean_relax", "N_rec_CI682_up_relax", "N_rec_CI682_dn_relax", "N_rec_CI95_up_relax", "N_rec_CI95_dn_relax", ...
            "N_max_data_strict", "N_max_data_relax", "chisq_over_N_strict", "chisq_over_N_relax", ...
            "sigmar_principal_strict", "sigmar_ptw_mean", "sigmar_mean", ...
            "sigmar_CI682_up", "sigmar_CI682_dn", "sigmar_CI95_up", "sigmar_CI95_dn", ...
            "sigmar_principal_relax", "sigmar_ptw_mean_relax", "sigmar_mean_relax", ...
            "sigmar_CI682_up_relax", "sigmar_CI682_dn_relax", "sigmar_CI95_up_relax", "sigmar_CI95_dn_relax", ...
            "N_rec_principal_noisy", "sigmar_principal_noisy", ...
            "lambda_strict", "lambda_relaxed", "lambda_noisy", "lambda_type", "NUM_SAMPLES", "eps_neg_penalty", ...
            "xbj_bin", "s_bin", "use_real_data", "data_type", "mscheme", "run_file", ...
            "use_low_Q_cut", "use_high_Q_cut", "q_cut", ...
            "-nocompression","-v7")
    end
end
