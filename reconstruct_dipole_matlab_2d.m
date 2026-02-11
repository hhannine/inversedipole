%
% inversedipole reconstruction implementation written by 
%       H. Hänninen, University of Jyväskylä
% in collaboration with
%       H. Schluter, University of Helsinki
% 
% Copyright 2026


% Include dependencies: AIRToolsII, Regtools
addpath(genpath('./dependencies'))
addpath(genpath("C:\Users\hana_\My Drive\Postdoc MathPhys\Project 2 - Inverse dipole LO\HenriAnttiPaperv2"))

close all
clear all
% parp = gcp;


function [ L ] = get_l_2d_modified( nr,nx,dr,dx )
    % Computes the L-matrix of a 2-dimensional problem
    % Supports varying matrix dimensions to work with varying number of datapoints.
    % nr = r_grid size
    % nx = number of xbj bins
    % dr = order of derivative operator in r (verify!)
    % dx =order of derivative operator in xbj (verify!)
    Lr = get_l(nr,dr);
    Lx = get_l(nx,dx);
    L = [kron(speye(nx),Lr);kron(Lx,speye(nr))];
    L = qr(L,0);
    L = L(1:end-1,:);
end


function chisq = calc_chisq(sigmar_rec, b_data, b_errs)
    chisq_vec = (sigmar_rec - b_data).^2 ./ b_errs.^2;
    chisq = sum(chisq_vec) / (length(chisq_vec)-1);
end


function a = calc_array_CI(arr)
    % Determine arr mean and confidence intervals
    p_tails_95 = [0.025, 0.975];
    p_tails_682 = [0.159, 0.841];
    rm_outliers = rmoutliers(arr, "percentiles", [1.5 98.5]);
    pd = fitdist(rm_outliers,'Kernel','Kernel','epanechnikov');
    arr_icdf_vals_95 = icdf(pd, p_tails_95);
    arr_icdf_vals_682 = icdf(pd, p_tails_682);
    a = [mean(pd), arr_icdf_vals_682(1), arr_icdf_vals_682(2), arr_icdf_vals_95(1), arr_icdf_vals_95(2)];
end


function dip_props = calc_dipole_prop_variance(dip_data_arr, r_grid, global_N_max)
    % Q_s: sampling saturation scale distribution for the reconstructions
    % N_max(r_max) UQ. Sampling the maximum allows to solve for the position of the maximum as well (and its variance)
    % each return variable is a list of [mean, +-1 std, +-2 std]
    NUM_SAMPLES = length(dip_data_arr(1,:));
    array_rec_sample_QsSig0distrib_data = zeros([NUM_SAMPLES,4]); % product with number of rec methods to compare for Q_s?
    parfor i=1:NUM_SAMPLES
        % TODO NEEDS TO BE SPLIT INTO XBJ BINS BY R_GRID LENGTH!
        stacked_rec_dip_i = dip_data_arr(:,i)';
        % [Nmax_dip_i, max_i] = max(rec_dip_i);
        % r_max_i = r_grid(max_i);
        interp_dip_i = makima(r_grid, rec_dip_i);
        % intrpfun_i = @(r) ppval(interp_dip_i,r)/Nmax_dip_i - 1 + exp(-0.5);
        intrpfun_i = @(r) ppval(interp_dip_i,r)/global_N_max - 1 + exp(-0.5);
        rs_i = fzero(intrpfun_i, 2);
        Qs_i = sqrt(2)/rs_i;
        array_rec_sample_QsSig0distrib_data(i,:) = [Nmax_dip_i, r_max_i, rs_i, Qs_i^2];
    end
    Nmax_dip = calc_array_CI(array_rec_sample_QsSig0distrib_data(:,1));
    r_max = calc_array_CI(array_rec_sample_QsSig0distrib_data(:,2));
    rs = calc_array_CI(array_rec_sample_QsSig0distrib_data(:,3));
    Qs = calc_array_CI(array_rec_sample_QsSig0distrib_data(:,4));
    dip_props = [Nmax_dip, r_max, rs, Qs];
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RUN VARIABLES
closure_testing = false;
use_real_data = true;

gev_to_mb = 1/2.56819;
s_bins = [318.1, 300.3, 251.5]; % 224.9 bin had no viable xbj bins at all
s_bin = s_bins(1);
s_str = "s" + string(s_bin);

% r_steps = 256; % first paper grid size
r_steps = 128; % LOG step
% r_steps = 65; % LOG step
r_steps_str = strcat("r_steps",int2str(r_steps));
ref_r_steps = 384; % beta1
ref_r_steps_str = strcat("r_steps",int2str(ref_r_steps));
% r_grid_type = "conventional";
% r_grid_type = "_new_";
r_grid_type = "_newv2_";

% forward operator data files
data_path = './export_hera_data_2d/';
data_files = dir(fullfile(data_path,'*.mat'));
data_type = "dis_inclusive"; % vs. dis_charm, dis_bottom, diff_dis_inclusive
data_name_key = "heraII_filtered";
% ref_data_name = "bayesMV4-strict_Q_cuts";
ref_data_name = "bayesMV4-wide_Q_cuts";
ref_data_name_key = "heraII_reference_dipoles_filtered_"+ref_data_name;

quark_mass_schemes = [
        "standard",
        "standard_light",
        "pole",
        "mqMpole",
        "mqmq",
        "mqMcharm",
        "mqMbottom",
        "mqMW",
    ];
ref_fit_mscheme = quark_mass_schemes(2);
% mscheme = quark_mass_schemes(1); % standard scheme with charm and bottom
% mscheme = quark_mass_schemes(2); % CKM reference fit mass scheme is light only
% mscheme = quark_mass_schemes(3);
% mscheme = quark_mass_schemes(4); % mqMpole, the more accurate alternative to 'standard'
% mscheme = quark_mass_schemes(5);
mscheme = quark_mass_schemes(6); % charm scale as the standard choice?
% mscheme = quark_mass_schemes(7); % bottom scale. n=10 prefers this over charm
% mscheme = quark_mass_schemes(8); % W boson mass scale for high Q^2?

lambda_type = "lambdaSRN"; % strict+relaxed+noisy
lam1 = 1:0.1:9.9;
% lambda_noisy = [lam1*6e-4,lam1*1e-3]; % testing for VERY noisy / peak ridge
lambda_noisy = [lam1*9e-4,lam1*1e-3]; % testing for noisy
lambda_relaxed = [lam1*1e-3, lam1*1e-2, lam1*1e-1]; 
lambda_strict = [lam1*2e-3, lam1*1e-2, lam1*1e-1];
lambda_t2 = [lam1*10e-2]; % This might be close for TIKH2 at 0.02??

% eps_neg_penalty=1e-2; % strict default?
% eps_neg_penalty=1e-4; %
eps_neg_penalty=0;
% Default target for chi^2 goodness of fit for the reconstruction
chi_goal = 1.0;

%           1  2  3   4  5   6  7  8   9  10  11  12  13  14  15
% q_cuts = [1.5, 4, 5, 9, 12, 15, 22, 22, 22, 22, 22, 20, 30, 90, 100]; % bins selected to cut small-r peak
q_cuts = 5*ones(15); % cut at Q^2=2 like the LO Bayesian fits
use_low_Q_cut = false;
% use_low_Q_cut = true;

q_high=50; % compare with 45~50 used by the LO Bayesian fits?
use_high_Q_cut = false;
% use_high_Q_cut = true;

save2file = false;
% save2file = true;
if save2file == true
    % n0 = 1;
    % nend = length(all_xbj_bins);
    plotting = false;
else
    plotting = true;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% MAIN RECONSTRUCTION: 2D, full dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Reading input discretized forward operator file:
["Loading file with:", data_name_key, mscheme, r_steps_str, s_str]
for k = 1:numel(data_files)
    fname = data_files(k).name;
    if (contains(fname, data_name_key) && contains(fname, mscheme) && contains(fname, r_steps_str) && contains(fname, r_grid_type) && contains(fname, s_str))
        run_file = fname;
    end
end
if run_file
    load(strcat(data_path, run_file))
else
    ["FILE NOT FOUND? with:", data_name_key, mscheme, r_steps_str, s_str]
    return
end
format long
rng(69,"twister");
r_grid(end) = [];
A = forward_op_A;
data_sigmar_rcs = data_sigmar_rcs; % rcs format: qsq, xbj, y, sqrt_s, sigmar, sig_err, theory
b_hera = data_sigmar_rcs(:,5);
b_errs = data_sigmar_rcs(:,6);
qsq_data_vals = data_sigmar_rcs(:,1);
x_data_vals = data_sigmar_rcs(:,2);
[xcounts, xbj_bins] = groupcounts(x_data_vals);
xbj_grid = xbj_bins';
[qcounts, qsq_bins] = groupcounts(qsq_data_vals);
qsq_grid = qsq_bins';

ref_dip_loaded = false;
ct_groundtruth_loaded = false;
N_fit = [];
sigmar_ref_dipole = [];

small_x_only = true;
if small_x_only
    %%%% TRIMMING DATA FOR SMALL XBJ
    % xbj_bins(14) % 0.013
    Ntrim = sum(xcounts(1:14));
    % x_data_vals(xend) % check end value
    A = A(1:Ntrim,1:(r_steps-1)*14); %TODO NEED TO TRIM THIS HORIZONTALLY AS WELL: nr*Ntrim columns
    b_hera = b_hera(1:Ntrim);
    b_errs = b_errs(1:Ntrim);
    qsq_data_vals = qsq_data_vals(1:Ntrim);
    x_data_vals = x_data_vals(1:Ntrim);
    [xcounts, xbj_bins] = groupcounts(x_data_vals);
    xbj_grid = xbj_bins';
    [qcounts, qsq_bins] = groupcounts(qsq_data_vals);
    qsq_grid = qsq_bins';

    [size(A), size(b_hera), size(b_errs)]
end

if closure_testing==false && false
    % Load reference dipole to have a comparison for the real data reconstruction.
    for k = 1:numel(data_files)
        fname = data_files(k).name;
        if (contains(fname, ref_data_name_key) && contains(fname, ref_fit_mscheme) && contains(fname, ref_r_steps_str) && contains(fname, s_str))
            ref_file = fname;
        end
    end
    ref_dip_data = load(strcat(data_path, ref_file));
    ref_qsq_vals = ref_dip_data.qsq_vals;
    % ref_real_sigma0 = 37.0628; % MV4 refit, in GeV^-2
    ref_dipole = ref_dip_data.discrete_dipole_N';
    N_fit = ref_dipole;
    ref_r_grid = ref_dip_data.r_grid;
    ref_r_grid(end) = [];
    sigmar_ref_dipole = ref_dip_data.sigmar_theory';
    ref_dip_bins_b_hera = ref_dip_data.sigmar_vals';
    ref_dip_bins_b_errs = ref_dip_data.sigmar_errs';
    ref_dip_loaded = true;

    chisq_vec_ref_dip = (sigmar_ref_dipole - ref_dip_bins_b_hera).^2 ./ ref_dip_bins_b_errs.^2;
    chisq_over_ref_dip = sum(chisq_vec_ref_dip) / length(chisq_vec_ref_dip);
    if chisq_over_ref_dip>1
        chi_goal = 1;
    else
        chi_goal = chisq_over_ref_dip;
    end
end

% testing Q binning -- Careful, limiting upper limit reduces the number of points, and the reconstrctuion can break with too few points!
q_cut = 0; % no cut low or high
if use_low_Q_cut
    if q2vals(1) > q_cuts(xi)
        "Smallest Q^2 point larger than cut!"
    end
    q_cut = q_cuts(xi);
    qn = find(q2vals>q_cut,1,'first');
    q2vals = q2vals(qn:length(q2vals)) % sub-bump disappears with a cut at 18 GeV?
    A = A(qn:length(q2vals)+qn-1,:);
    b_hera = b_hera(qn:length(q2vals)+qn-1);
    b_errs = b_errs(qn:length(q2vals)+qn-1);
end
if use_high_Q_cut
    if q2vals(end) < q_high
        "highest Q^2 point lower than cut!"
    end
    eps_neg_penalty=1e10;
    q_cut = q_high;
    qn = find(q2vals<q_cut,1,'last');
    q2vals = q2vals(1:qn)
    A = A(1:qn,:);
    b_hera = b_hera(1:qn);
    b_errs = b_errs(1:qn);
end
% [data_name_key, s_bin, r_steps, use_real_data, mscheme, length(q2vals), min(q2vals), mean(sqrt(q2vals)), mean(q2vals), length(all_xbj_bins)]
[data_name_key, s_bin, r_steps, use_real_data, mscheme]


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main 2D reconstruction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nr=r_steps;
nx=length(xbj_bins);
L1=get_l_2d_modified(nr-1,nx,1,1);

% first order derivative operator
[UU,sm,XX] = cgsvd(A,L1);
% second order derivative operator
% [UU2,sm2,XX2] = cgsvd(A,L2);
% principal reconstruction to actual data points
X_tikh_principal = tikhonov(UU,sm,XX,b_hera,lambda_strict);
X_tikh_principal_relax = tikhonov(UU,sm,XX,b_hera,lambda_relaxed);
X_tikh_principal_noisy = tikhonov(UU,sm,XX,b_hera,lambda_noisy);
% runner up methods
% X_tikh2 = tikhonov(UU2,sm2,XX2,b_hera,lambda_t2);
options.lbound = 0;
kmax1 = 2500;
kmax2 = 5000;
kmaxcim2 = kmax2;
% xC = cimmino(A,b_hera,1:kmax,[],options);
% xK = kaczmarz(A,b_hera,1:kmax,[],options);
% xC1 = PCimmino2(A,b_hera,kmax1,L1,W1);
% xK1 = PKaczmarz2(A,b_hera,kmax1,L1,W1);
% xC2 = PCimmino2(A,b_hera,kmaxcim2,L2,W2);
% xK2 = PKaczmarz2(A,b_hera,kmax2,L2,W2);

errtik_p = zeros(size(lambda_strict));
errtik_prel = zeros(size(lambda_relaxed));
errtik_pnoisy = zeros(size(lambda_noisy));
errtik2 = zeros(size(lambda_t2));

for i = 1:length(lambda_strict)
    % errtik_p(i) = norm((b_hera-A*X_tikh_principal(:,i)))/norm(b_hera) + eps_neg_penalty*(1-sign(min(X_tikh_principal(:,i))));
    chisq_v = (A*X_tikh_principal(:,i) - b_hera).^2 ./ b_errs.^2;
    errtik_p(i) = abs(sum(chisq_v) / (length(chisq_v)-1) - chi_goal);
end
for i = 1:length(lambda_relaxed)
    errtik_prel(i) = norm((b_hera-A*X_tikh_principal_relax(:,i)))/norm(b_hera) + eps_neg_penalty*(1-sign(min(X_tikh_principal_relax(:,i))));
end
for i = 1:length(lambda_noisy)
    errtik_pnoisy(i) = norm((b_hera-A*X_tikh_principal_noisy(:,i)))/norm(b_hera) + eps_neg_penalty*(1-sign(min(X_tikh_principal_noisy(:,i))));
end
% for i = i:length(lambda_t2)
%     % errtik2(i) = norm((b_hera-A*X_tikh2(:,i)))/norm(b_hera) + 10000*eps_neg_penalty*(1-sign(min(X_tikh2(:,i))));
%     chisq_v = (A*X_tikh2(:,i) - b_hera).^2 ./ b_errs.^2;
%     errtik2(i) = abs(sum(chisq_v) / (length(chisq_v)-1) - chi_goal);
% end
[mp,mIp]=min(errtik_p);
[mpr,mIpr]=min(errtik_prel);
[mpn,mIpn]=min(errtik_pnoisy);
lambda_unity_chisq = lambda_strict(mIp);
rec_dip_principal_strict = X_tikh_principal(:,mIp);
rec_dip_principal_relax = X_tikh_principal_relax(:,mIpr);
rec_dip_principal_noisy = X_tikh_principal_noisy(:,mIpn);
sigmar_principal_strict = A*rec_dip_principal_strict;
sigmar_principal_relax = A*rec_dip_principal_relax;
sigmar_principal_noisy = A*rec_dip_principal_noisy;

init_testing = false;
% init_testing = true;
if init_testing
    % Xim = reshape(X_tikh_principal(:,mIp),[],nx);
    Xim = reshape(X_tikh_principal_relax(:,mIpr),[],nx);
    % Xim = reshape(X_tikh_principal_noisy(:,mIpn),[],nx);
    % logData = Xim .* log( abs( Xim ) );
    figure(1)
    % imagesc(xbj_grid, r_grid, Xim)
    surf(xbj_grid, r_grid, Xim)
    % surf(xbj_grid, r_grid, abs(Xim))
    hAx=gca;
    set(hAx,{'XScale','YScale'},{'log','log'});
    % set(hAx,{'XScale','YScale','ZScale'},{'log','log','log'})
    % return

    size(sigmar_principal_strict)
    figure(2)

    sigmar_vec = sigmar_principal_strict;
    % sigmar_vec = sigmar_principal_relax;
    % sigmar_vec = sigmar_principal_noisy;
    plot3(x_data_vals, qsq_data_vals, sigmar_vec)
    hAx=gca;
    set(hAx,{'XScale','YScale'},{'log','log'});
end

% comparison recs
% [m2,mI2]=min(errtik2);
% rec_dip_principal_tik2 = X_tikh2(:,mI2);
% rec_cimmino1 = xC1(:,end);
% rec_kacz1 = xK1(:,end);
% % rec_kacz2 = xK2(:,end);
% sigmar_tikh2 = A*rec_dip_principal_tik2;
% sigmar_cimmino1 = A*rec_cimmino1;
% sigmar_kacz1 = A*rec_kacz1;
% % sigmar_kacz2 = A*rec_kacz2;

% CHI^2 TEST for the principal rec's agreement with the real data
chisq_over_N_strict = calc_chisq(sigmar_principal_strict, b_hera, b_errs);
chisq_over_N_relax = calc_chisq(sigmar_principal_relax, b_hera, b_errs);
chisq_over_N_noisy = calc_chisq(sigmar_principal_noisy, b_hera, b_errs);

if ref_dip_loaded
    chisq_vec_ref_dip = (sigmar_ref_dipole - ref_dip_bins_b_hera).^2 ./ ref_dip_bins_b_errs.^2;
    chisq_over_ref_dip = sum(chisq_vec_ref_dip) / length(chisq_vec_ref_dip);
end
if ref_dip_loaded && closure_testing==false
    % chisq_data = [chisq_over_N_strict, chisq_over_N_relax, chisq_over_N_noisy, chisq_over_Nt2, chisq_over_Nk1, chisq_over_Ncim1, chisq_over_ref_dip]
    chisq_data = [chisq_over_N_strict, chisq_over_N_relax, chisq_over_N_noisy, chisq_over_ref_dip]
else
    % chisq_data = [chisq_over_N_strict, chisq_over_N_relax, chisq_over_N_noisy, chisq_over_Nt2, chisq_over_Nk1, chisq_over_Ncim1]
    chisq_data = ["chisq over N:", chisq_over_N_strict, chisq_over_N_relax, chisq_over_N_noisy]
end

[max(rec_dip_principal_strict), max(rec_dip_principal_relax), max(rec_dip_principal_noisy)]

% return    % Testing main rec


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% BOOTSTRAPPING UNCERTAINTIES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% bootstrapping reconstruction uncertainties with Uncorrelated Experiment Uncertainties.
array_over_dataset_samples_dipole_recs = [];
array_over_dataset_samples_sigmar = [];
array_over_dataset_samples_dipole_recs_rel = [];
array_over_dataset_samples_sigmar_rel = [];
array_over_dataset_samples_dipole_recs_tik2 = [];
array_over_dataset_samples_sigmar_tik2 = [];
NUM_SAMPLES = 50;
parfor j=1:NUM_SAMPLES
    err = b_errs.*randn(length(b_hera),1);
    b = b_hera + err;

    X_tikh = tikhonov(UU,sm,XX,b,lambda_strict);
    errtik = zeros(size(lambda_strict));
    X_tikh_rel = tikhonov(UU,sm,XX,b,lambda_relaxed);
    errtik_rel = zeros(size(lambda_relaxed));
    % X_tikh2 = tikhonov(UU2,sm2,XX2,b,lambda_t2);
    % errtik2 = zeros(size(lambda_t2));
    
    for i = 1:length(lambda_strict)
        % errtik(i) = abs(norm((b-A*X_tikh(:,i))/(b_errs))-1);
        % errtik(i) = norm((b-A*X_tikh(:,i)))/norm(b) + eps_neg_penalty*(1-sign(min(X_tikh(:,i))));
        chisq_v = (A*X_tikh(:,i) - b).^2 ./ err.^2;
        errtik(i) = abs(sum(chisq_v) / (length(chisq_v)-1) - chi_goal);
    end
    for i = 1:length(lambda_relaxed)
        errtik_rel(i) = norm((b-A*X_tikh_rel(:,i)))/norm(b) + eps_neg_penalty*(1-sign(min(X_tikh_rel(:,i))));
    end
    % for i = i:length(lambda_t2)
    %     % errtik2(i) = norm((b_hera-A*X_tikh2(:,i)))/norm(b_hera) + 10000*eps_neg_penalty*(1-sign(min(X_tikh2(:,i))));
    %     chisq_v = (A*X_tikh2(:,i) - b).^2 ./ err.^2;
    %     errtik2(i) = abs(sum(chisq_v) / (length(chisq_v)-1) - chi_goal);
    % end
    [m,mI]=min(errtik);
    [m_rel,mI_rel]=min(errtik_rel);
    % [m2,mI2]=min(errtik2);

    rec_dip = X_tikh(:,mI);
    array_over_dataset_samples_dipole_recs(:,j) = rec_dip;
    array_over_dataset_samples_sigmar(:,j) = A*rec_dip;
    rec_dip_rel = X_tikh_rel(:,mI_rel);
    array_over_dataset_samples_dipole_recs_rel(:,j) = rec_dip_rel;
    array_over_dataset_samples_sigmar_rel(:,j) = A*rec_dip_rel;
    % rec_dip_tikh2 = X_tikh2(:,mI2);
    % array_over_dataset_samples_dipole_recs_tik2(:,j) = rec_dip_tikh2;
    % array_over_dataset_samples_sigmar_tik2(:,j) = A*rec_dip_tikh2;
end % rec loop over dataset samples ends here

% Reconstruction statistics / bootstrapping for pointwise distributions
dataset_sample_pdfs = zeros([length(rec_dip_principal_strict),5]);
dataset_sample_pdfs_sigmar = zeros([length(b_hera),5]);
dataset_sample_pdfs_rel = zeros([length(rec_dip_principal_strict),5]);
dataset_sample_pdfs_sigmar_rel = zeros([length(b_hera),5]);
% dataset_sample_pdfs_tik2 = zeros([length(q2vals),5]);
% dataset_sample_pdfs_sigmar_tik2 = zeros([length(q2vals),5]);
p_tails_95 = [0.025, 0.975];
p_tails_682 = [0.159, 0.841];
parfor j=1:length(rec_dip_principal_strict)
    %princip
    rec_dips_at_rj = array_over_dataset_samples_dipole_recs(j,:)';
    dip_rm_outliers = rmoutliers(rec_dips_at_rj, "percentiles", [1.5 98.5]);
    pd = fitdist(dip_rm_outliers,'Kernel','Kernel','epanechnikov'); % see available methods with 'methods(pd)'
    dip_icdf_vals_95 = icdf(pd, p_tails_95);
    dip_icdf_vals_682 = icdf(pd, p_tails_682);
    %relax
    rec_dips_at_rj_rel = array_over_dataset_samples_dipole_recs_rel(j,:)';
    dip_rm_outliers_rel = rmoutliers(rec_dips_at_rj_rel, "percentiles", [1.5 98.5]);
    pd_rel = fitdist(dip_rm_outliers_rel,'Kernel','Kernel','epanechnikov'); % see available methods with 'methods(pd)'
    dip_icdf_vals_95_rel = icdf(pd_rel, p_tails_95);
    dip_icdf_vals_682_rel = icdf(pd_rel, p_tails_682);
    %tikh2
    % rec_dips_at_rj_tik2 = array_over_dataset_samples_dipole_recs_tik2(j,:)';
    % dip_rm_outliers_tik2 = rmoutliers(rec_dips_at_rj_tik2, "percentiles", [1.5 98.5]);
    % pd_tik2 = fitdist(dip_rm_outliers_tik2,'Kernel','Kernel','epanechnikov'); % see available methods with 'methods(pd)'
    % dip_icdf_vals_95_tik2 = icdf(pd_tik2, p_tails_95);
    % dip_icdf_vals_682_tik2 = icdf(pd_tik2, p_tails_682);

    dataset_sample_pdfs(j,:) = [mean(pd), dip_icdf_vals_682(1), dip_icdf_vals_682(2), dip_icdf_vals_95(1), dip_icdf_vals_95(2)];
    dataset_sample_pdfs_rel(j,:) = [mean(pd_rel), dip_icdf_vals_682_rel(1), dip_icdf_vals_682_rel(2), dip_icdf_vals_95_rel(1), dip_icdf_vals_95_rel(2)];
    % dataset_sample_pdfs_tik2(j,:) = [mean(pd_tik2), dip_icdf_vals_682_tik2(1), dip_icdf_vals_682_tik2(2), dip_icdf_vals_95_tik2(1), dip_icdf_vals_95_tik2(2)];
end
parfor j=1:length(b_hera)
    %princip
    sigmar_at_Qj = array_over_dataset_samples_sigmar(j,:)';
    pd_sig = fitdist(sigmar_at_Qj,'Kernel','Kernel','epanechnikov');
    sig_icdf_vals_95 = icdf(pd_sig, p_tails_95);
    sig_icdf_vals_682 = icdf(pd_sig, p_tails_682);
    dataset_sample_pdfs_sigmar(j,:) = [mean(pd_sig), sig_icdf_vals_682(1), sig_icdf_vals_682(2), sig_icdf_vals_95(1), sig_icdf_vals_95(2)];
    %relax
    sigmar_at_Qj_rel = array_over_dataset_samples_sigmar_rel(j,:)';
    pd_sig_rel = fitdist(sigmar_at_Qj_rel,'Kernel','Kernel','epanechnikov');
    sig_icdf_vals_95_rel = icdf(pd_sig_rel, p_tails_95);
    sig_icdf_vals_682_rel = icdf(pd_sig_rel, p_tails_682);
    dataset_sample_pdfs_sigmar_rel(j,:) = [mean(pd_sig_rel), sig_icdf_vals_682_rel(1), sig_icdf_vals_682_rel(2), sig_icdf_vals_95_rel(1), sig_icdf_vals_95_rel(2)];
    %tikh2
    % sigmar_at_Qj_tik2 = array_over_dataset_samples_sigmar_tik2(j,:)';
    % pd_sig_tik2 = fitdist(sigmar_at_Qj_tik2,'Kernel','Kernel','epanechnikov');
    % sig_icdf_vals_95_tik2 = icdf(pd_sig_tik2, p_tails_95);
    % sig_icdf_vals_682_tik2 = icdf(pd_sig_tik2, p_tails_682);
    % dataset_sample_pdfs_sigmar_tik2(j,:) = [mean(pd_sig_tik2), sig_icdf_vals_682_tik2(1), sig_icdf_vals_682_tik2(2), sig_icdf_vals_95_tik2(1), sig_icdf_vals_95_tik2(2)];
end

% Reconstructions, means and confidence intervals
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
% N_rec_principal_tik2 = rec_dip_principal_tik2;
% N_rec_ptw_mean_tik2 = dataset_sample_pdfs_tik2(:,1);
% N_rec_CI682_up_tik2 = dataset_sample_pdfs_tik2(:,3); % 68.2% confidence interval upper limit
% N_rec_CI682_dn_tik2 = dataset_sample_pdfs_tik2(:,2); % 68.2% c.i. lower limit
% N_rec_CI95_up_tik2 = dataset_sample_pdfs_tik2(:,5); % 95% confidence interval upper limit
% N_rec_CI95_dn_tik2 = dataset_sample_pdfs_tik2(:,4); % 95% c.i. lower limit
N_rec_principal_noisy = rec_dip_principal_noisy;
if min(N_rec_principal) < 0
    "Negative principal rec."
    % [M,I] = min(N_rec_principal);
    % ["Negative principal", M, M/max(N_rec_principal), r_grid(I), mean(b_errs./b_hera), max(b_errs./b_hera)]
end

% TODO SELECTING MAXIMUM FROM THE WHOLE SET TAKES THE MAX OF RECONSTRUCTION IN THE FULL DATA SET
% IS THIS GOOD OR BAD? good: saturation scale is defined by a single maximum of N(r,x).
%                      bad:  doesn't work naively: Nmax - N(r, x_i) won't go to zero for other x_i \neq x_maxN
%                   compromise: define r_s (and Q_s) by the scale r_s where the dipole Nmax_i - N(r,x_i) reaches the ~40% of the global maximum?
%                               This will push the saturation scale r_s to very large for large x_i, since the peak is low, as it should?
%   - To implement this compromise definition we need:
%       - the global maximum N_max = N(r,x)
%       - 'local' maxima at each x_i, which is used to define S(r) = N_locmax_i - N(r,x)

% Sigma02: reconstruction maxima and C.I.s
[max(N_rec_principal), max(N_rec_principal_relax), max(N_rec_principal_noisy)]
[N_max_strict, max_i] = max(rec_dip_principal_strict);
[N_max_ptw_mean, max_i_mean] = max(N_rec_ptw_mean);
[N_max_relax, max_i_rel] = max(rec_dip_principal_relax);
[N_max_noisy, max_i_noisy] = max(rec_dip_principal_noisy);
r_max_scale = find( r_grid > 6, 1 ); % this should perhaps be determined by the vanishing of the fwd operator at large r?
% [N_max_tik2_candid1, max_i_t2] = max(N_rec_principal_tik2);
% N_max_tik2_candid2 = N_rec_principal_tik2(r_max_scale);
% if r_grid(max_i_t2) > 20
%     % Peak at too large r, or at RMAX, choose inferred peak at top end of sensitivity at the end of the intermediate regime
%     N_max_tik2 = N_max_tik2_candid2;
% else
%     N_max_tik2 = N_max_tik2_candid1;
% end
r_Nmax_strict = r_grid(rem(max_i,length(xbj_grid)));
r_Nmax_rel = r_grid(rem(max_i_rel,length(xbj_grid)));
r_Nmax_noisy = r_grid(rem(max_i_noisy,length(xbj_grid)));
N_max_strict_ci = [N_rec_CI682_dn(max_i), N_rec_CI682_up(max_i), N_rec_CI95_dn(max_i), N_rec_CI95_up(max_i)];
N_max_relax_ci = [N_rec_CI682_dn_relax(max_i_rel), N_rec_CI682_up_relax(max_i_rel), N_rec_CI95_dn_relax(max_i_rel), N_rec_CI95_up_relax(max_i_rel)];
N_max_data_strict = [N_max_strict, r_Nmax_strict, N_max_strict_ci];
N_max_data_relax = [N_max_relax, r_Nmax_rel, N_max_relax_ci];


% SATURATION SCALE
% TODO FIX SATURATION (Global N_max needed from above)
"TODO: FIX SATURATION"
% dip_props_strict = calc_dipole_prop_variance(array_over_dataset_samples_dipole_recs, r_grid); % returns Nmax, rmax, rs, Qs as [mean, dn, up, dn2, up2]
% dip_props_rel = calc_dipole_prop_variance(array_over_dataset_samples_dipole_recs_rel, r_grid);
% dip_props_tik2 = calc_dipole_prop_variance(array_over_dataset_samples_dipole_recs_tik2, r_grid);
ref_dip_props = [];
if ref_dip_loaded
    % TODO NEEDS TO BE UPDATED FOR THE STACKED PROBLEM
    dip_interp_ref = makima(ref_r_grid, ref_dipole);
    % dip_interp_ref = makima(r_grid, ref_dipole);
    fun_dip_qs_ref = @(r) ppval(dip_interp_ref,r)/max(ref_dipole) - 1 + 0.606530659712;
    rs_ref = fzero(fun_dip_qs_ref, 2);
    Qs_ref = sqrt(2)/rs_ref;
    ref_dip_props = [max(ref_dipole), r_grid(end), rs_ref, Qs_ref^2];
end

% SIGMA_R REDUCED CROSS SECTIONS
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% PLOTTING
    
    


    

plotting = true;
plot_relax = false;
plot_tik2 = false;
plot_comp_methods = false;
if plotting
    figure(1) % rec_princip vs. mean reconstruction vs. ground truth
    Xim_p = reshape(X_tikh_principal(:,mIp),[],nx);
    Xim_r = reshape(X_tikh_principal_relax(:,mIpr),[],nx);
    Xim_n = reshape(X_tikh_principal_noisy(:,mIpn),[],nx);
    surf(xbj_grid, r_grid, Xim_p, "DisplayName", "principal")
    % set(hAx,{'XScale','YScale','ZScale'},{'log','log','log'})
    hold on
    Xim_p_CI682_up = reshape(N_rec_CI682_up,[],nx);
    Xim_p_CI682_dn = reshape(N_rec_CI682_dn,[],nx);
    surf(xbj_grid, r_grid, Xim_p_CI682_up,'FaceLighting','gouraud',...
    'MeshStyle','column',...
    'SpecularColorReflectance',0,...
    'SpecularExponent',5,...
    'SpecularStrength',0.2,...
    'DiffuseStrength',1,...
    'AmbientStrength',0.4,...
    'AlignVertexCenters','on',...
    'LineWidth',0.2,...
    'FaceAlpha',0.2,...
    'FaceColor',[0.07 0.6 1],...
    'EdgeAlpha',0.2);
    surf(xbj_grid, r_grid, Xim_p_CI682_dn,'FaceLighting','gouraud',...
    'MeshStyle','column',...
    'SpecularColorReflectance',0,...
    'SpecularExponent',5,...
    'SpecularStrength',0.2,...
    'DiffuseStrength',1,...
    'AmbientStrength',0.4,...
    'AlignVertexCenters','on',...
    'LineWidth',0.2,...
    'FaceAlpha',0.2,...
    'FaceColor',[1 0.6 0.07],...
    'EdgeAlpha',0.2);
    hAx=gca;
    set(hAx,{'XScale','YScale'},{'log','log'});
    hold off

    figure(2)
    surf(xbj_grid, r_grid, Xim_r, "DisplayName", "relax")
    hAx=gca;
    set(hAx,{'XScale','YScale'},{'log','log'});

    figure(3)
    surf(xbj_grid, r_grid, Xim_n, "DisplayName", "noisy")
    hAx=gca;
    set(hAx,{'XScale','YScale'},{'log','log'});

    figure(4)
    surf(xbj_grid, r_grid, Xim_r./Xim_p)
    hold on
    surf(xbj_grid, r_grid, Xim_n./Xim_p,'FaceLighting','gouraud',...
    'MeshStyle','column',...
    'SpecularColorReflectance',0,...
    'SpecularExponent',5,...
    'SpecularStrength',0.2,...
    'DiffuseStrength',1,...
    'AmbientStrength',0.4,...
    'AlignVertexCenters','on',...
    'LineWidth',0.2,...
    'FaceAlpha',0.2,...
    'FaceColor',[0.07 0.6 1],...
    'EdgeAlpha',0.2);
    hAx=gca;
    set(hAx,{'XScale','YScale'},{'log','log'});
    ylim([0.3 inf]);
    zlim([0 4]);
    clim([0 3]);
    hold off
        
    % Reduced cross section data comparison plot(s)
    % All the xbj bins are reconstructed simultaneously, so there's ~15 bins of data to compare.
    % basic: throw all in the same plot? -> going to be a huge mess?
    % grid_plot: plot each bin separately and show a grid of comparisons? Would be quite good but will be more complex to do.
    figure(5)
    sigmar_vec = sigmar_principal_strict;
    % sigmar_vec = sigmar_principal_relax;
    % sigmar_vec = sigmar_principal_noisy;
    plot3(x_data_vals, qsq_data_vals, sigmar_vec, 'ob', 'DisplayName',"sigmar-principal")
    hAx=gca;
    set(hAx,{'XScale','YScale'},{'log','log'});
    hold on
    plot3(x_data_vals, qsq_data_vals, b_hera,'xr', 'DisplayName',"hera")
    % errorbar(q2vals', b_hera, b_errs, '')
    plot3([x_data_vals,x_data_vals]', [qsq_data_vals,qsq_data_vals]', [-b_errs,b_errs]'+b_hera', '-r')

    hold off
end

% EXPORTING RESULTS
if save2file
    if use_ref_dipole_data==false
        % reconstructing from HERA data
        rec_type_str = "rec_hera_data";
        if ref_dip_loaded==false
            % no ref available
            exp_ref_str = "norefdip";
        else
            % ref available
            exp_ref_str = ref_data_name;
        end
    elseif use_ref_dipole_data
        % reconstructing from simulated reference dipole data (with the same measurement points and errors as HERA)
        rec_type_str = "rec_refdip_data";
        exp_ref_str = ref_data_name; % fit name is in here.
    end

    data_name = rec_type_str+"_"+data_type+"_"+s_str;
    q_cut_str = "no_Q_cut_";
    if use_low_Q_cut == true
        q_cut_str = "cut_low_Q_";
    elseif use_high_Q_cut == true
        q_cut_str = "cut_high_Q_";
    end
    flavor_string = mscheme;
    name = [data_name, '_', flavor_string, '_', lambda_type, '_', q_cut_str];
    recon_path = "./reconstructions_hera_uq/";
    f_exp_reconst = strjoin([recon_path 'hera_recon_uq_' r_steps '_' name '_xbj' xbj_bin '.mat'],"")
    N_reconst = N_rec_principal;
    N_rec_one_std_up = N_rec_CI682_up; % N_rec + std
    N_rec_one_std_dn = N_rec_CI682_dn; % N_rec - std
    b_from_reconst = sigmar_principal_strict; % prescription of the data by the reconstructred dipole.

    % TODO
    % Re-design data that is saved to file

    save(f_exp_reconst, ...
        "r_grid", "r_steps", "q2vals", ...
        "N_reconst", "N_rec_one_std_up", "N_rec_one_std_dn", ...
        "N_fit", "sigmar_ref_dipole", ...
        "b_from_reconst", "b_hera", "b_errs", ...
        "N_rec_principal", "N_rec_ptw_mean", "N_rec_CI682_up", "N_rec_CI682_dn", "N_rec_CI95_up", "N_rec_CI95_dn", ...
        "N_rec_principal_relax", "N_rec_ptw_mean_relax", "N_rec_CI682_up_relax", "N_rec_CI682_dn_relax", "N_rec_CI95_up_relax", "N_rec_CI95_dn_relax", ...
        "N_max_data_strict", "N_max_data_relax", "chisq_over_N_strict", "chisq_over_N_relax", ...
        "dip_props_strict", "dip_props_rel", "dip_props_tik2", "ref_dip_props", ...
        "sigmar_principal_strict", "sigmar_ptw_mean", "sigmar_mean", ...
        "sigmar_CI682_up", "sigmar_CI682_dn", "sigmar_CI95_up", "sigmar_CI95_dn", ...
        "sigmar_principal_relax", "sigmar_ptw_mean_relax", "sigmar_mean_relax", ...
        "sigmar_CI682_up_relax", "sigmar_CI682_dn_relax", "sigmar_CI95_up_relax", "sigmar_CI95_dn_relax", ...
        "N_rec_principal_noisy", "sigmar_principal_noisy", ...
        "chisq_data", ...
        "lambda_strict", "lambda_relaxed", "lambda_noisy", "lambda_type", "NUM_SAMPLES", "eps_neg_penalty", ...
        "xbj_bin", "s_bin", "use_real_data", "data_type", "mscheme", "run_file", ...
        "use_low_Q_cut", "use_high_Q_cut", "q_cut", ...
        "-nocompression","-v7")
end
