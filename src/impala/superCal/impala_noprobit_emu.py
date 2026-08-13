####################################
####################################
"""Impala Model Fit Definitions"""
####################################
####################################

###############
### Imports ###
###############

from collections import defaultdict
from math import floor, log, sqrt

import numpy as np
from numpy.linalg import cholesky, slogdet
from numpy.random import normal
from scipy.special import erf, erfinv, gammaln, multigammaln

from ..physics import PTW_goodparam

np.seterr(under="ignore")

# no probit tranform for hierarchical and DP versions

###############################################################
### CalibSetup Class for Initializing the Calibration Model ###
###############################################################


def is_valid_mapping(theta_inds, s2_inds):
    error_to_x = defaultdict(set)

    for x, e in zip(theta_inds, s2_inds):
        error_to_x[e].add(x)

    return all(len(xs) == 1 for xs in error_to_x.values())


class CalibSetup:
    """
    Structure for storing calibration experimental data, likelihood, discrepancy, etc.
    Includes the following methods:

    * addVecExperiments
    * setTemperatureLadder
    * setMCMC
    * setHierPriors
    * setClusterPriors

    """

    def __init__(self, bounds, constraint_func="bounds", theta0_start=None):
        """
        Initialize the structure for storing data, models, etc.

        :param bounds: dictionary where keys are parameter names and items are tuples that are the
            lower and upper bounds for the parameters
        :param constraint_func: a function that takes a dictionary of parameter combinations as well
            as the dictionary of bounds and (these should have matching keys) and returns a vector
            of 1s and 0s with 1s where the parameter combinations meet the constraint. Alternatively,
            if no constraints other than bounds exist, passing "bounds" for this argument will use
            the proper constraint function.
        :param theta0_start: (optional) an array with dimension ntemps x # parameters with initial
            values of the calibration parameters. Each parameter should already be rescaled to be within
            [0,1] following the bounds provided. ntemps should match the number of temperatures provided in
            setTemperatureLadder later on. If not specified, the sampler initialization is randomly chosen.

        """

        self.nexp = 0  # Number of independent emulators
        self.ys = []
        self.y_lens = []
        self.models = []
        self.tl = np.array(1.0)
        self.itl = 1 / self.tl
        self.bounds = bounds  # should be a dict so we can use parameter names
        self.bounds_mat = np.array(list(bounds.values()))
        self.p = bounds.__len__()
        if constraint_func is None:
            constraint_func = lambda *x: True
        if constraint_func == "bounds":
            constraint_func = cf_bounds
        # self.checkConstraints = constraint_func  ## see wrapper below (maintains run-script compatibility with earlier impala versions)
        self._constraint_func = constraint_func
        self.nmcmc = 10000
        self.nburn = 5000
        self.thin = 5
        self.decor = 100
        self.ntemps = 1
        self.sd_est = []
        self.s2_df = []
        self.ig_a = []
        self.ig_b = []
        self.wt = []
        self.sd_lower = []
        self.sd_upper = []
        self.s2_ind = []
        self.s2_exp_ind = []
        self.ns2 = []
        self.ny_s2 = []
        self.ntheta = []
        self.theta_ind = []
        self.nswap = 5
        self.s2_prior_kern = []
        self.constants = None
        self.theta0_start = None  # optional
        self.theta_start = None  # optional

    def checkConstraints(self, x, *args):
        """Calls the constraint function set by the user. Argument x contains the parameters to be checked
        using self.bounds and self.constants (required if parameters are kept constant by the user.
        Optional variables *args can be used to override the latter two, i.e. if len(args)=1,
        only self.bounds is overridden with the user-provided input, if len(args)=2, both are.
        Further additional arguments are currently ignored."""
        bounds = self.bounds
        consts = self.constants
        lenargs = len(args)
        if lenargs >= 1:
            bounds = args[0]
        if lenargs >= 2:
            consts = args[1]
        return self._constraint_func(x, bounds, consts)

    def addVecExperiments(
        self,
        yobs,
        model,
        sd_est,
        s2_df,
        s2_ind,
        wt=None,
        sd_lower=None,
        sd_upper=None,
        meas_error_cor=None,
        theta_ind=None,
        D=None,
        discrep_tau=1,
    ):
        """
        Add an experiment (really a data/model combination), or a set of experiments for which
        model prediction of the quantity of interest is vectorized.

        :param yobs: a vector (numpy array) of observed data
        :param model: a class with an eval method.  The eval method should include arguments
            parmat (a dictionary with keys matching those in the bounds dictionary, and items
            that are parameter combinations), pool (logical indicating whether this is a
            calculation for a pooled model or a hierarchical model), and nugget (logical
            indicating whether prediction is done including a nugget term) and should return
            the model evaluations at the parameter combinations.  If this is an emulator with
            posterior samples, a step method can also be included
        :param sd_est: a list or numpy array of initial values for observation noise standard deviation, len(sd_est) = number of separately-estimated s2 values
        :param s2_df: a list or numpy array of initial values for s2 Inverse Gamma prior degrees of freedom (s2_df = 0; Half-Cauchy prior), same structure as sd_est
        :param s2_ind: a list or numpy array of indices for s2 value associated with each element of yobs, len(s2_ind) = len(yobs), max(s2_ind)+1 = len(sd_est)
        :param wt: (optional) a list or numpy array of indices for weight value associated with each element of yobs, len(wt) = len(yobs)
        :param sd_lower: (optional) a list or numpy array of indices for measurement error lower bounds, len(sd_est) = number of separately-estimated s2 values
        :param sd_upper: (optional) a list or numpy array of indices for measurement error upper bounds, len(sd_est) = number of separately-estimated s2 values
        :param meas_error_cor: (optional) correlation matrix for observation measurement errors, default = independent
        :param theta_ind: a list or numpy array of indices for theta_i associated with each element of yobs (usually, indexes experiments), len(theta_ind) = len(yobs)
        :param D: (optional) numpy array containing basis functions for discrepancy, possibly including intercept. D.shape = (length of yobs, number of bases)
        :param discrep_tau: (optional) fixed prior variance for discrepancy basis coefficients (discrepancy = D @ discrep_vars, discrep_vars ~ N(0,discrep_tau))
        """
        # if theta_ind specified, s2_ind is?
        yobs = np.array(yobs)
        sd_est = np.array(sd_est)
        s2_df = np.array(s2_df)
        s2_ind = np.array(s2_ind)
        if len(yobs.shape) != 1:
            raise ValueError("len(yobs.shape) should be 1")
        if len(sd_est.shape) != 1:
            raise ValueError("len(sd_est.shape) should be 1")
        if len(s2_df.shape) != 1:
            raise ValueError("len(s2_df.shape) should be 1")
        if s2_ind.dtype != np.int_:
            raise ValueError(f"s2_ind.dtype should be {np.int_}")
        if len(yobs) != len(s2_ind):
            raise ValueError("len(yobs) and len(s2_ind) should be the same")
        self.ys.append(np.array(yobs))
        self.y_lens.append(len(yobs))
        if (theta_ind is not None) and (
            not is_valid_mapping(np.array(theta_ind), np.array(s2_ind))
        ):
            print(
                "Warning: Cannot have multiple thetas with shared measurement error."
            )
        if theta_ind is None:
            theta_ind = [0] * len(yobs)

        theta_ind = np.array(theta_ind)

        if wt is not None:
            wt = np.array(wt)
        else:
            wt = np.repeat(1, len(yobs)).flatten()  # equal weights

        self.wt.append(wt)
        model.exp_ind = theta_ind  # past versions of impala had trouble with custom emulators + hierarchical clustered calibration. Fixed bug here.

        if sd_lower is not None:
            sd_lower = np.array(sd_lower)
            sd_upper = np.array(sd_upper)
            self.sd_lower.append(sd_lower)
            self.sd_upper.append(sd_upper)

        self.theta_ind.append(theta_ind)
        self.ntheta.append(len(set(theta_ind)))
        model.yobs = np.array(yobs)

        # model.meas_error_cor = np.eye(len(yobs)) # this doesn't work when ntheta>1
        if meas_error_cor is not None:
            model.meas_error_cor = meas_error_cor

        if D is not None:
            model.D = D
            model.nd = D.shape[1]
            model.discrep_tau = discrep_tau

        self.models.append(model)
        self.constants = self.models[0].constants
        self.nexp += 1
        self.sd_est.append(sd_est)
        self.s2_df.append(s2_df)
        self.ig_a.append(s2_df / 2)
        self.ig_b.append(s2_df / 2 * sd_est**2)
        self.s2_ind.append(s2_ind)
        self.s2_exp_ind.append(list(range(sd_est.size)))
        self.ns2.append(sd_est.size)
        vec = np.empty(sd_est.size)
        for i in range(len(vec)):
            vec[i] = np.sum(s2_ind == i)
        self.ny_s2.append(vec)
        if np.any(s2_df == 0):
            self.s2_prior_kern.append(ldhc_kern)
        else:
            self.s2_prior_kern.append(ldig_kern)

    def setTemperatureLadder(self, temperature_ladder, start_temper=1000):
        """
        Define an array of temperatures to use for parallel tempering

        :param temperature_ladder : numpy array of increasing temperatures all above 1, e.g., 1.05**np.arange(50)
        :param start_temper : (optional) MCnC iteration at which to start the parallel tempering, default = 1000
        """
        self.tl = temperature_ladder
        self.itl = 1 / self.tl
        self.ntemps = len(self.tl)
        self.nswap_per = floor(self.ntemps // 2)
        self.start_temper = start_temper

    def setMCMC(
        self,
        nmcmc,
        nburn=0,
        thin=1,
        decor=100,
        start_var_theta=1e-8,
        start_tau_theta=0.0,
        start_var_ls2=1e-5,
        start_tau_ls2=0.0,
        start_adapt_iter=300,
    ):
        """
        Define properties of MCMC algorithm

        :param nmcmc : total number of MCMC iterations, including burn-in
        :param nburn : deprecated, no longer used
        :param thin : deprecated, no longer used
        :param decor : currently not used
        :param start_var_theta : (optional) initial variance of adaptive MCMC proposal distributions for theta.
            Can be increased from default if posterior samples of theta are stuck at a single value across many iterations
        :param start_tau_theta : (optional) np.exp(start_tau_theta) is the initial scaling factor for the adaptive MCMC proposal covariance for theta.
            Can be kept at default for most users.
        :param start_var_ls2 : (optional) initial variance of adaptive MCMC proposal distributions for log(s2), i.e. the log of the observation error/noise standard deviation.
            Can be increased from default if posterior samples of theta are stuck at a single value across many iterations
        :param start_tau_ls2 : (optional) np.exp(start_tau_ls2) is the initial scaling factor for the adaptive MCMC proposal covariance for log(s2).
            Can be kept at default for most users.
        :param start_adapt_iter : (optional) MCMC iteration at which to start adapting the MCMC proposal distributions.
            Can be left as default for most users.
        """
        self.nmcmc = nmcmc
        self.nburn = nburn
        self.thin = thin
        self.decor = decor
        self.start_var_theta = start_var_theta
        self.start_tau_theta = start_tau_theta
        self.start_var_ls2 = start_var_ls2
        self.start_tau_ls2 = start_tau_ls2
        self.start_adapt_iter = start_adapt_iter

    def setHierPriors(
        self,
        theta0_prior_mean,
        theta0_prior_cov,
        Sigma0_prior_df,
        Sigma0_prior_scale,
    ):
        """
        Define hierachical model hyperparameters, where theta_i ~ N(theta0, Sigma0)
        with priors (1) theta0~N(mean=theta0_prior_mean,covariance=theta0_prior_cov) and (2) Sigma0~InverseWishart(V=Sigma0_prior_scale,m=Sigma0_prior_df)
        where prior E(Sigma0)=V/(m-p-1) and Var(Sigma0)ii = 2 (Vii)^2/(((m-p-1)^2)*(m-p-3))

        :param theta0_prior_mean : numpy array of length self.p containing initial values for calibration parameter theta0, usually np.repeat(0.5, self.p)
        :param theta0_prior_cov : numpy array (self.p by self.p) containing the prior covariance for theta0, usually np.eye(self.p)*user_defined_prior_variance
        :param Sigma0_prior_df :  prior degrees of freedom for the prior for Sigma0, at least 1 + self.p,
            where larger values generally indicate theta_i values closer to theta_0
        :param Sigma0_prior_scale : prior scale for the prior for Sigma0, where smaller values generally indicate theta_i values closer to theta_0
        """
        self.theta0_prior_mean = theta0_prior_mean
        self.theta0_prior_cov = theta0_prior_cov
        self.Sigma0_prior_df = Sigma0_prior_df
        self.Sigma0_prior_scale = Sigma0_prior_scale

    def setClusterPriors(
        self, nclustmax=None, eta_prior_shape=2, eta_prior_rate=0.1
    ):
        """
        Define clustered experiment model hyperparameters

        :param nclustmax : maximum number of unique theta values to estimate (i.e., maximum number of clusters)
        :param eta_prior_shape : shape from the gamma prior for the DP concentration parameter eta
        :param eta_prior_rate :  rate from the gamma prior for the DP concentration parameter eta
        """
        if nclustmax is None:
            nclustmax = max(sum(self.ntheta), 10)
        self.nclustmax = nclustmax
        self.eta_prior_shape = eta_prior_shape
        self.eta_prior_rate = eta_prior_rate


########################
### Helper Functions ###
########################


def constraints_ptw(x, bounds, constants=None):
    """Checks if the given PTW parameter set is valid. Required input variables: parameters 'x' and their (calibration) bounds 'bounds;
    Any parameter the user chose to keep constant during a calibration goes into the optional variable 'constants' instead of x and bounds."""
    if constants is None:
        constants = {}
    y = constants | x
    good = PTW_goodparam(
        s0=y["s0"],
        sInf=y["sInf"],
        y0=y["y0"],
        yInf=y["yInf"],
        y1=y["y1"],
        y2=y["y2"],
        beta=y["beta"],
    )
    for k, v in bounds.items():
        good = good * (x[k] < v[1]) * (x[k] > v[0])
    return good


def cf_bounds(x, bounds, constants=None):
    """default for bounds checking, variable 'constants' is not used here and present only to have a consistent api."""
    if constants is None:
        constants = {}
    k = next(iter(bounds.keys()))
    good = x[k] < bounds[k][1]
    for k, v in bounds.items():
        good = good * (x[k] < v[1]) * (x[k] > v[0])
    return good


def normalize(x, bounds):
    """Normalize to 0-1 scale"""
    return (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def unnormalize(z, bounds):
    """Inverse of normalize"""
    return z * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]


def probit(x):
    """Probit Transformation: For x in (0,1), y in (-inf,inf)"""
    return np.sqrt(2.0) * erfinv(2 * x - 1)


def invprobit(y):
    """Inverse Probit Transformation: For y in (-inf,inf), x in (0,1)"""
    return 0.5 * (1 + erf(y / np.sqrt(2.0)))


initfunc_probit = (
    np.random.normal
)  # if probit, then normal--if uniform, then uniform
initfunc_unif = np.random.uniform


def subset_dict(dd, idx):
    return {key: value[idx] for key, value in dd.items()}


def tran_probit(th, bounds, names):
    return dict(zip(names, unnormalize(invprobit(th), bounds).T))  # If probit
    # return dict(zip(names, unnormalize(th, bounds).T)) # If uniform


def tran_unif(th, bounds, names):
    return dict(zip(names, unnormalize(th, bounds).T))  # If uniform


def chol_sample(mean, cov):
    return mean + np.dot(
        np.linalg.cholesky(cov), np.random.standard_normal(mean.size)
    )


def chol_sample_1per(means, covs):
    return means + np.einsum(
        "tnpq,tnq->tnp", cholesky(covs), normal(size=means.shape)
    )


def chol_sample_nper(means, covs, n):
    return means + np.einsum(
        "ijk,ilk->ilj", cholesky(covs), normal(size=(*means.shape, n))
    )


def chol_sample_1per_constraints(
    means, covs, cf, bounds, consts, maxiter=1000000
):
    """
    Sample with constraints.  If fail constraints, resample.
    """
    assert isinstance(bounds, dict), (
        "please update your script calling chol_sample_1per_constraints(means, covs, cf, bounds, consts)!"
    )
    bounds_keys = bounds.keys()
    bounds_mat = np.array(list(bounds.values()))
    chols = cholesky(covs)
    cand = means + np.einsum("ijk,ik->ij", chols, normal(size=means.shape))
    good = cf(tran_unif(cand, bounds_mat, bounds_keys), bounds, consts)
    i = 1
    while np.any(np.logical_not(good)):
        if i > maxiter:
            raise ValueError(
                f"Failed to find samples that fulfill the constraints after {maxiter} iterations."
            )
        cand[np.where(np.logical_not(good))] = +means[
            np.logical_not(good)
        ] + np.einsum(
            "ijk,ik->ij",
            chols[np.logical_not(good)],
            normal(size=((np.logical_not(good)).sum(), means.shape[1])),
        )
        good[np.logical_not(good)] = cf(
            tran_unif(cand[np.logical_not(good)], bounds_mat, bounds_keys),
            bounds,
        )
        i += 1
    return cand


def chol_sample_nper_constraints(
    means, covs, n, cf, bounds, consts, maxiter=1000000
):
    """Sample with constraints.  If fail constraints, resample."""
    assert isinstance(bounds, dict), (
        "please update your script calling chol_sample_1per_constraints(means, covs, cf, bounds, consts)!"
    )
    bounds_keys = bounds.keys()
    bounds_mat = np.array(list(bounds.values()))
    chols = cholesky(covs)
    cand = means.reshape(means.shape[0], 1, means.shape[1]) + np.einsum(
        "ijk,ink->inj", chols, normal(size=(means.shape[0], n, means.shape[1]))
    )
    for i in range(cand.shape[0]):
        goodi = cf(tran_unif(cand[i], bounds_mat, bounds_keys), bounds, consts)
        j = 0
        while np.any(np.logical_not(goodi)):
            if j > maxiter:
                raise ValueError(
                    f"Failed to find samples that fulfill the constraints after {maxiter} iterations."
                )
            cand[i, np.where(np.logical_not(goodi))[0]] = +means[i] + np.einsum(
                "ik,nk->ni",
                chols[i],
                normal(size=((np.logical_not(goodi)).sum(), means.shape[1])),
            )
            goodi[np.where(np.logical_not(goodi))[0]] = cf(
                tran_unif(
                    cand[i, np.where(np.logical_not(goodi))[0]],
                    bounds_mat,
                    bounds_keys,
                ),
                bounds,
            )
            j += 1
    return cand


def cov_3d_pcm(arr, mean):
    """Covariance array from 3d Array (with pre-computed mean):
    arr = 3d Array (nSamp x nTemp x nCol)
    mean = 2d Array (nTemp x nCol)
    out = 3d Array (nTemp x nCol x nCol)
    """
    N = arr.shape[0]
    return np.einsum("kij,kil->ijl", arr - mean, arr - mean) / (N - 1)


def cov_4d_pcm(arr, mean):
    """Covariance Array from 4d Array (With pre-computed mean):
    arr = 4d array (nSamp x nTemp x nTheta x nCol)
    mean = 3d Array (nTemp x nCol)
    out = 4d Array (nTemp x nTheta x nCol x nCol)
    """
    N = arr.shape[0]
    return np.einsum("ktij,ktil->tijl", arr - mean, arr - mean) / (N - 1)


def cov_anyd_pcm(arr, mean):
    """Covariance Array from p dimensional Array (With pre-computed mean):
    arr = p-dim array (e.g., nSamp x nTemp x nTheta x nCol)
    mean = (p-1)-dim Array (e.g., nTemp x nCol)
    out = p-dim Array (nTemp x nTheta x nCol x nCol)
    """
    N = arr.shape[0]
    return np.einsum("...ij,...il->...ijl", arr - mean, arr - mean) / (N - 1)


def mvnorm_logpdf(x, mean, Prec, ldet):  # VALIDATED
    """
    # k = x.shape[-1]
    # part1 = -k * 0.5 * np.log(2 * np.pi) - 0.5 * ldet
    # x = x - mu
    # return part1 + np.squeeze(-x[..., None, :] @ Prec @ x[..., None] / 2)
    """
    ld = (
        -0.5 * x.shape[-1] * 1.8378770664093453
        - 0.5 * ldet
        - 0.5 * np.einsum("tm,mn,tn->t", x - mean, Prec, x - mean)
    )
    return ld


def mvnorm_logpdf_(x, mean, prec, ldet):  # VALIDATED
    """
    x = (ntemps, n_theta[i], k)
    mu = (ntemps[i])
    prec = (ntemps x k x k)
    ldet = (ntemps)
    """
    # m = np.repeat(mean.reshape(mean.shape[0], 1, mean.shape[1]), x.shape[1], 1)
    mean_reshape = (mean.shape[0], 1, mean.shape[1])
    ld = (
        -0.5 * x.shape[-1] * 1.8378770664093453
        - 0.5 * ldet.reshape(-1, 1)
        - 0.5
        * np.einsum(
            "tsm,tmn,tsn->ts",
            x - mean.reshape(mean_reshape),
            prec,
            x - mean.reshape(mean_reshape),
        )
    )
    return ld


def invwishart_logpdf(w, df, scale):  # VALIDATED
    """unnormalized logpdf of inverse wishart w given df and scale"""
    ld = (
        +0.5 * df * slogdet(scale)[1]
        - multigammaln(df / 2, scale.shape[-1])
        - 0.5 * df * scale.shape[-1] * log(2.0)
        - 0.5 * (df + w.shape[-1] + 1) * slogdet(w)[1]
        - 0.5
        * np.einsum(
            "...ii->...", np.einsum("ji,...ij->...ij", scale, np.linalg.inv(w))
        )
    )
    return ld


def invgamma_logpdf(s, alpha, beta):
    """log pdf of inverse gamma distribution -- Assume s = (n x p); alpha, beta = (p)"""
    ld = (
        +alpha * np.log(beta)
        - gammaln(alpha)
        - (alpha - 1) * np.log(s)
        - beta / s
    ).sum(axis=1)
    return ld


def gamma_logpdf(s, alpha, beta):
    """logpdf pf gamma distribution -- assume s = (n); alpha, beta  = 1"""
    ld = (
        +alpha * np.log(beta)
        - gammaln(alpha)
        + (alpha - 1) * np.log(s)
        - beta * s
    )
    return ld


def ldig_kern(x, a, b):  # ig
    return (-a - 1) * np.log(x) - b / x


def ldhc_kern(x, a, b):  # half cauchy
    return -np.log(x + 1)


class AMcov_pool:
    def __init__(
        self, ntemps, p, start_var=1e-4, start_adapt_iter=300, tau_start=0.0
    ):
        self.eps = 1.0e-12
        self.AM_SCALAR = 2.4**2 / p
        self.tau = np.repeat(tau_start, ntemps)
        self.S = np.empty([ntemps, p, p])
        self.S[:] = np.eye(p) * start_var
        self.cov = np.empty([ntemps, p, p])
        self.mu = np.empty([ntemps, p])
        self.ntemps = ntemps
        self.p = p
        self.start_adapt_iter = start_adapt_iter
        self.count_100 = np.zeros(ntemps, dtype=int)

    def update(self, x, m):
        if m > self.start_adapt_iter:
            self.mu += (x[m - 1] - self.mu) / m
            self.cov = +((m - 1) / m) * self.cov + (
                (m - 1) / (m * m)
            ) * np.einsum("ti,tj->tij", x[m - 1] - self.mu, x[m - 1] - self.mu)
            self.S = self.AM_SCALAR * np.einsum(
                "ijk,i->ijk",
                self.cov + np.eye(self.p) * self.eps,
                np.exp(self.tau),
            )
            # S   = cc * np.einsum('ijk,i->ijk', cov_3d_pcm(theta[:m], theta[:m].mean(axis = 0)) + np.eye(setup.p) * eps, np.exp(tau))

        elif m == self.start_adapt_iter:
            self.mu = x[:m].mean(axis=0)
            self.cov = cov_3d_pcm(x[:m], self.mu)
            self.S = self.AM_SCALAR * np.einsum(
                "ijk,i->ijk",
                self.cov + np.eye(self.p) * self.eps,
                np.exp(self.tau),
            )

    def update_tau(self, m):
        # diminishing adaptation based on acceptance rate for each temperature
        if (m % 100 == 0) and (m > self.start_adapt_iter):
            delta = min(0.5, 5 / sqrt(m + 1))
            self.tau[np.where(self.count_100 < 23)] = (
                self.tau[np.where(self.count_100 < 23)] - delta
            )
            self.tau[np.where(self.count_100 > 23)] = (
                self.tau[np.where(self.count_100 > 23)] + delta
            )
            self.count_100 *= 0
            # note, e^tau scales whole covariance matrix, so it shrinks covariance for inert inputs too much...need decor for those.

    def gen_cand(self, x, m):
        x_cand = +x[m - 1] + np.einsum(
            "ijk,ik->ij", cholesky(self.S), normal(size=(self.ntemps, self.p))
        )
        return x_cand


class AMcov_hier:
    def __init__(
        self,
        nexp,
        ntheta,
        ntemps,
        p,
        start_var=1e-4,
        start_adapt_iter=300,
        tau_start=0.0,
    ):  # ntheta is a vector of length nexp
        self.eps = 1.0e-12
        self.AM_SCALAR = 2.4**2 / p
        self.tau = [
            tau_start * np.ones((ntemps, ntheta[i])) for i in range(nexp)
        ]
        self.S = [np.empty((ntemps, ntheta[i], p, p)) for i in range(nexp)]
        for i in range(nexp):
            self.S[i][:] = np.eye(p) * start_var
        self.cov = [np.empty((ntemps, ntheta[i], p, p)) for i in range(nexp)]
        self.mu = [np.empty((ntemps, ntheta[i], p)) for i in range(nexp)]
        self.nexp = nexp
        self.ntemps = ntemps
        self.p = p
        self.start_adapt_iter = start_adapt_iter
        self.count_100 = [np.zeros((ntemps, ntheta[i])) for i in range(nexp)]

    def update(
        self, x, m
    ):  # called in mth iteration, so latest value is x[i][m-1]
        if m > self.start_adapt_iter:
            for i in range(self.nexp):
                self.mu[i] += (x[i][m - 1] - self.mu[i]) / m
                self.cov[i][:] = +((m - 1) / m) * self.cov[i] + (
                    (m - 1) / (m * m)
                ) * np.einsum(
                    "tej,tel->tejl",
                    x[i][m - 1] - self.mu[i],
                    x[i][m - 1] - self.mu[i],
                )
                self.S[i] = self.AM_SCALAR * np.einsum(
                    "tejl,te->tejl",
                    self.cov[i] + np.eye(self.p) * self.eps,
                    np.exp(self.tau[i]),
                )

        elif m == self.start_adapt_iter:
            for i in range(self.nexp):
                self.mu[i][:] = x[i][:m].mean(axis=0)
                # self.mu[i][:]  = x[i].mean(axis = 0)
                self.cov[i][:] = cov_4d_pcm(x[i][:m], self.mu[i])
                self.S[i][:] = self.AM_SCALAR * np.einsum(
                    "tejl,te->tejl",
                    self.cov[i] + np.eye(self.p) * self.eps,
                    np.exp(self.tau[i]),
                )

    def update_tau(self, m):
        # diminishing adaptation based on acceptance rate for each temperature
        if (m % 100 == 0) and (m > self.start_adapt_iter):
            delta = min(0.5, 5 / np.sqrt(m + 1))
            for i in range(self.nexp):
                self.tau[i][self.count_100[i] < 23] -= delta
                self.tau[i][self.count_100[i] > 23] += delta
                self.count_100[i] *= 0

    def gen_cand(self, x, m):
        x_cand = [
            chol_sample_1per(x[i][m - 1], self.S[i]) for i in range(self.nexp)
        ]
        return x_cand
