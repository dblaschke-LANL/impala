####################################
####################################
"""Impala Pooled calibration"""
####################################
####################################

###############
### Imports ###
###############

import time
from collections import namedtuple
from multiprocessing import Pool

import numpy as np
from numpy.random import uniform

from .impala_noprobit_emu import AMcov_pool, initfunc_unif, tran_unif
from .pbar import pbar

np.seterr(under="ignore")

OutCalibPool = namedtuple(
    "OutCalibPool",
    "theta s2 count count_s2 count_decor cov_theta_cand cov_ls2_cand pred_curr discrep_vars llik theta_native",
)


# @profile
def calibPool(setup):
    """
    Perform pooled calibration with expanded capabilities, still undergoing testing
    Some changes include:, allowing weights, allowing custom initializations, adding truncated gibbs sampling for measurement errors
    """
    t0 = time.time()
    theta = np.empty([setup.nmcmc, setup.ntemps, setup.p])
    log_s2 = [
        np.ones([setup.nmcmc, setup.ntemps, setup.ns2[i]])
        for i in range(setup.nexp)
    ]
    for i in range(setup.nexp):
        log_s2[i][0] = np.log(setup.sd_est[i] ** 2)
    # s2_vec_curr = [s2[i][0,:,setup.s2_ind[i]] for i in range(setup.nexp)]
    s2_ind_mat = [
        (setup.s2_ind[i][:, None] == range(setup.ns2[i]))
        for i in range(setup.nexp)
    ]
    theta_start0 = initfunc_unif(size=[setup.ntemps, setup.p])
    good = setup.checkConstraints(
        tran_unif(theta_start0, setup.bounds_mat, setup.bounds.keys())
    )
    while np.any(np.logical_not(good)):
        theta_start0[np.where(np.logical_not(good))] = initfunc_unif(
            size=[(np.logical_not(good)).sum(), setup.p]
        )
        good[np.where(np.logical_not(good))] = setup.checkConstraints(
            tran_unif(
                theta_start0[np.where(np.logical_not(good))],
                setup.bounds_mat,
                setup.bounds.keys(),
            )
        )
    theta[0] = theta_start0

    s2_which_mat = [
        [
            np.where(s2_ind_mat[i][:, j])[0]
            for j in range(s2_ind_mat[i].shape[1])
        ]
        for i in range(setup.nexp)
    ]

    wt_mat = [None] * setup.nexp
    for i in range(setup.nexp):
        wt_mat[i] = setup.wt[i]
        if (
            np.any(
                np.asarray([
                    len(np.unique(wt_mat[i][s2_which_mat[i][j]]))
                    for j in range(len(s2_which_mat[i]))
                ])
            )
            != 1
        ) and (setup.models[i].s2 == "gibbs"):
            setup.models[i].s2 = "fix"
            print(
                "Gibbs sampling for s2 only valid if weights are the same for all observations with same s2. Reverting to fixed s2. "
            )

    itl_mat = [  # matrix of temperatures for use with alpha calculation--to skip nested for loops.
        (np.ones((setup.ns2[i], setup.ntemps)) * setup.itl).T
        for i in range(setup.nexp)
    ]

    pred_curr = [None] * setup.nexp
    # sse_curr = np.empty([setup.ntemps, setup.nexp])
    llik_curr = np.empty([setup.nexp, setup.ntemps])
    # dev_sq = [np.empty((setup.ntemps, setup.ns2[i])) for i in range(setup.nexp)]
    marg_lik_cov_curr = [None] * setup.nexp
    for i in range(setup.nexp):
        marg_lik_cov_curr[i] = [None] * setup.ntemps
        for t in range(setup.ntemps):
            marg_lik_cov_curr[i][t] = setup.models[i].lik_cov_inv(
                np.exp(log_s2[i][0, t, setup.s2_ind[i]])[setup.s2_ind[i]],
                wt_mat[i],
                setup.s2_ind[i],
            )
            # ask around: is list of lists lookup slow?? ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    llik_curr[:] = 0.0
    for i in range(setup.nexp):
        pred_curr[i] = setup.models[i].eval(
            tran_unif(theta[0], setup.bounds_mat, setup.bounds.keys()),
            pool=True,
        )
        # sse_curr[:, i] = np.sum((pred_curr[i] - setup.ys[i]) ** 2 / s2_vec_curr[i].T, 1)
        # ((pred_curr[i] - setup.ys[i])**2 @ s2_ind_mat[i] / s2[i][0]).sum(axis = 1)
        for t in range(setup.ntemps):
            llik_curr[i, t] = setup.models[i].llik(
                setup.ys[i], pred_curr[i][t], marg_lik_cov_curr[i][t], wt_mat[i]
            )

    # eps  = 1.0e-13
    # tau  = np.repeat(-4.0, setup.ntemps)
    # AM_const   = 2.4**2/setup.p
    # S    = np.empty([setup.ntemps, setup.p, setup.p])
    # S[:] = np.eye(setup.p)*1e-6
    # cov  = np.empty([setup.ntemps, setup.p, setup.p])
    # mu   = np.empty([setup.ntemps, setup.p])

    cov_theta_cand = AMcov_pool(
        ntemps=setup.ntemps,
        p=setup.p,
        start_var=setup.start_var_theta,
        tau_start=setup.start_tau_theta,
        start_adapt_iter=setup.start_adapt_iter,
    )
    cov_ls2_cand = [
        AMcov_pool(
            ntemps=setup.ntemps,
            p=setup.ns2[i],
            start_var=setup.start_var_ls2,
            tau_start=setup.start_tau_ls2,
            start_adapt_iter=setup.start_adapt_iter,
        )
        for i in range(setup.nexp)
    ]

    count = np.zeros([setup.ntemps, setup.ntemps], dtype=int)
    count_s2 = np.zeros([setup.nexp, setup.ntemps], dtype=int)
    count_decor = np.zeros([setup.p, setup.ntemps], dtype=int)
    # count_100 = np.zeros(setup.ntemps, dtype = int)

    pred_cand = [_.copy() for _ in pred_curr]
    discrep_curr = [_ * 0.0 for _ in pred_curr]
    discrep_vars = [
        np.zeros([setup.nmcmc, setup.ntemps, setup.models[i].nd])
        for i in range(setup.nexp)
    ]

    llik_cand = llik_curr.copy()

    alpha = np.ones(setup.ntemps) * (-np.inf)
    alpha_s2 = np.ones([setup.nexp, setup.ntemps]) * (-np.inf)
    sw_alpha = np.zeros(setup.nswap_per)

    llik = np.empty(setup.nmcmc)

    ## start MCMC
    for m in pbar(range(1, setup.nmcmc)):
        theta[m] = theta[
            m - 1
        ].copy()  # current set to previous, will change if accepted
        for i in range(setup.nexp):
            log_s2[i][m] = log_s2[i][m - 1].copy()
            if setup.models[i].nd > 0:  # update discrepancy
                for t in range(setup.ntemps):
                    discrep_vars[i][m][t] = setup.models[i].discrep_sample(
                        setup.ys[i],
                        pred_curr[i][t],
                        marg_lik_cov_curr[i][t],
                        setup.itl[t],
                        wt_mat[i],
                    )
                    discrep_curr[i][t] = (
                        setup.models[i].D @ discrep_vars[i][m][t]
                    )

            setup.models[i].step()
            if setup.models[i].stochastic:  # update emulator
                pred_curr[i] = setup.models[i].eval(
                    tran_unif(theta[m], setup.bounds_mat, setup.bounds.keys()),
                    pool=True,
                )
            if setup.models[i].nd > 0 or setup.models[i].stochastic:
                for t in range(setup.ntemps):
                    llik_curr[i, t] = setup.models[i].llik(
                        setup.ys[i] - discrep_curr[i][t],
                        pred_curr[i][t],
                        marg_lik_cov_curr[i][t],
                        wt_mat[i],
                    )

        ##################
        ### Draw Theta ###
        ##################

        cov_theta_cand.update(theta, m)

        # ------------------------------------------------------------------------------------------
        # generate proposal
        theta_cand = cov_theta_cand.gen_cand(theta, m)
        good_values = setup.checkConstraints(
            tran_unif(theta_cand, setup.bounds_mat, setup.bounds.keys())
        )
        # ------------------------------------------------------------------------------------------
        # get predictions and SSE
        pred_cand = [_.copy() for _ in pred_curr]
        llik_cand[:] = llik_curr.copy()
        if np.any(good_values):
            llik_cand[:, good_values] = 0.0
            for i in range(setup.nexp):
                pred_cand[i][good_values] = setup.models[i].eval(
                    tran_unif(
                        theta_cand[good_values],
                        setup.bounds_mat,
                        setup.bounds.keys(),
                    ),
                    pool=True,
                )
                for t in range(setup.ntemps):
                    llik_cand[i, t] = setup.models[i].llik(
                        setup.ys[i] - discrep_curr[i][t],
                        pred_cand[i][t],
                        marg_lik_cov_curr[i][t],
                        wt_mat[i],
                    )

        llik_diff = (llik_cand.sum(axis=0) - llik_curr.sum(axis=0))[
            good_values
        ]  # sum over experiments
        # ------------------------------------------------------------------------------------------
        # for each temperature, accept or reject
        alpha[:] = -np.inf
        alpha[good_values] = setup.itl[good_values] * (llik_diff)
        for t in np.where(np.log(uniform(size=setup.ntemps)) < alpha)[0]:
            theta[m, t] = theta_cand[t].copy()
            count[t, t] += 1
            for i in range(setup.nexp):
                llik_curr[i, t] = llik_cand[i, t].copy()
                pred_curr[i][t] = pred_cand[i][t].copy()
            cov_theta_cand.count_100[t] += 1

        cov_theta_cand.update_tau(m)

        ##########################
        ### Decorrelation Step ###
        ##########################
        if m % setup.decor == 0:
            for k in range(setup.p):
                theta_cand = theta[m].copy()
                theta_cand[:, k] = initfunc_unif(
                    size=setup.ntemps
                )  # independence proposal, will vectorize of columns
                good_values = setup.checkConstraints(
                    tran_unif(theta_cand, setup.bounds_mat, setup.bounds.keys())
                )
                pred_cand = [_.copy() for _ in pred_curr]
                llik_cand[:] = llik_curr.copy()

                if np.any(good_values):
                    llik_cand[:, good_values] = 0.0
                    for i in range(setup.nexp):
                        pred_cand[i][good_values] = setup.models[i].eval(
                            tran_unif(
                                theta_cand[
                                    good_values
                                ],  # .repeat(setup.ns2[i], axis = 0),
                                setup.bounds_mat,
                                setup.bounds.keys(),
                            ),
                            pool=True,
                        )
                        for t in range(setup.ntemps):
                            llik_cand[i, t] = setup.models[i].llik(
                                setup.ys[i] - discrep_curr[i][t],
                                pred_cand[i][t],
                                marg_lik_cov_curr[i][t],
                                wt_mat[i],
                            )  # (((pred_cand[i] - setup.ys[i])**2 @ s2_ind_mat[i]) / s2[i][m-1]).sum(axis = 1)

                alpha[:] = -np.inf
                # tsq_diff = 0.#((theta_cand * theta_cand).sum(axis = 1) - (theta[m] * theta[m]).sum(axis = 1))[good_values]
                llik_diff = (llik_cand.sum(axis=0) - llik_curr.sum(axis=0))[
                    good_values
                ]
                alpha[good_values] = (
                    setup.itl[good_values] * (llik_diff)
                )  # + tsq_diff) + 0.5 * tsq_diff # last is for proposal, since this is an independence sampler step
                for t in np.where(np.log(uniform(size=setup.ntemps)) < alpha)[
                    0
                ]:
                    theta[m, t, k] = theta_cand[t, k].copy()
                    count_decor[k, t] += 1
                    for i in range(setup.nexp):
                        pred_curr[i][t] = pred_cand[i][t].copy()
                        llik_curr[i, t] = llik_cand[i, t].copy()

        # ------------------------------------------------------------------------------------------
        #################
        ### Update s2 ###
        #################
        for i in range(setup.nexp):
            if setup.models[i].s2 == "gibbs":
                ## gibbs update s2
                dev_sq = (pred_curr[i] - setup.ys[i]) ** 2 @ s2_ind_mat[
                    i
                ]  # squared deviations
                for t in range(setup.ntemps):
                    log_s2[i][m][t] = np.log(
                        1
                        / np.random.gamma(
                            (
                                itl_mat[i][t]
                                * (setup.ny_s2[i] / 2 + setup.ig_a[i] + 1)
                                - 1
                            ),
                            (
                                1
                                / (
                                    itl_mat[i][t]
                                    * (setup.ig_b[i] + dev_sq[t].flatten() / 2)
                                )
                            ),
                        )
                    )
                    marg_lik_cov_curr[i][t] = setup.models[i].lik_cov_inv(
                        np.exp(log_s2[i][m][t])[setup.s2_ind[i]],
                        wt_mat[i],
                        setup.s2_ind[i],
                    )
                    llik_curr[i, t] = setup.models[i].llik(
                        setup.ys[i] - discrep_curr[i][t],
                        pred_curr[i][t],
                        marg_lik_cov_curr[i][t],
                        wt_mat[i],
                    )

            elif setup.models[i].s2 == "gibbs_trunc":
                dev_sq = (pred_curr[i] - setup.ys[i]) ** 2 @ s2_ind_mat[
                    i
                ]  # squared deviations
                for t in range(setup.ntemps):
                    log_s2[i][m][t] = np.log(
                        1
                        / np.random.gamma(
                            (
                                itl_mat[i][t]
                                * (setup.ny_s2[i] / 2 + setup.ig_a[i] + 1)
                                - 1
                            ),
                            (
                                1
                                / (
                                    itl_mat[i][t]
                                    * (setup.ig_b[i] + dev_sq[t].flatten() / 2)
                                )
                            ),
                        )
                    )
                    s2_is_valid = (
                        log_s2[i][m][t] >= np.log(setup.sd_lower[i] ** 2)
                    ) * (log_s2[i][m][t] <= np.log(setup.sd_upper[i] ** 2))
                    ct = 0
                    while np.any(~s2_is_valid):
                        sub = np.where(~s2_is_valid)
                        log_s2[i][m][t][sub] = np.log(
                            1
                            / np.random.gamma(
                                (
                                    itl_mat[i][t][sub]
                                    * (
                                        setup.ny_s2[i][sub] / 2
                                        + setup.ig_a[i][sub]
                                        + 1
                                    )
                                    - 1
                                ),
                                (
                                    1
                                    / (
                                        itl_mat[i][t][sub]
                                        * (
                                            setup.ig_b[i][sub]
                                            + dev_sq[t].flatten()[sub] / 2
                                        )
                                    )
                                ),
                            )
                        )
                        s2_is_valid = (
                            log_s2[i][m][t] >= np.log(setup.sd_lower[i] ** 2)
                        ) * (log_s2[i][m][t] <= np.log(setup.sd_upper[i] ** 2))
                        ct = ct + 1
                        if ct >= 50:
                            log_s2[i][m][t][
                                log_s2[i][m][t] < np.log(setup.sd_lower[i] ** 2)
                            ] = np.log(setup.sd_lower[i] ** 2)[
                                log_s2[i][m][t] < np.log(setup.sd_lower[i] ** 2)
                            ]
                            log_s2[i][m][t][
                                log_s2[i][m][t] > np.log(setup.sd_upper[i] ** 2)
                            ] = np.log(setup.sd_upper[i] ** 2)[
                                log_s2[i][m][t] > np.log(setup.sd_upper[i] ** 2)
                            ]
                            s2_is_valid = (
                                log_s2[i][m][t]
                                >= np.log(setup.sd_lower[i] ** 2)
                            ) * (
                                log_s2[i][m][t]
                                <= np.log(setup.sd_upper[i] ** 2)
                            )

                    marg_lik_cov_curr[i][t] = setup.models[i].lik_cov_inv(
                        np.exp(log_s2[i][m][t])[setup.s2_ind[i]],
                        wt_mat[i],
                        setup.s2_ind[i],
                    )
                    llik_curr[i, t] = setup.models[i].llik(
                        setup.ys[i] - discrep_curr[i][t],
                        pred_curr[i][t],
                        marg_lik_cov_curr[i][t],
                        wt_mat[i],
                    )

            elif setup.models[i].s2 == "fix":
                log_s2[i][m] = np.log(setup.sd_est[i] ** 2)

            else:
                ## M-H update s2
                # NOTE: there is something wrong with this...with no tempering, 10 kolski experiments,
                # reasonable priors, s2 can diverge for some experiments (not a random walk, has weird patterns).
                # This seems to be because of the joint update, but is strange.  Could be that individual updates
                # would make it go away, but it shouldn't be there anyway.

                cov_ls2_cand[i].update(log_s2[i], m)
                ls2_candi = cov_ls2_cand[i].gen_cand(log_s2[i], m)

                llik_candi = np.zeros(setup.ntemps)
                marg_lik_cov_candi = [None] * setup.ntemps
                for t in range(setup.ntemps):
                    marg_lik_cov_candi[t] = setup.models[i].lik_cov_inv(
                        np.exp(ls2_candi[t])[setup.s2_ind[i]],
                        wt_mat[i],
                        setup.s2_ind[i],
                    )
                    llik_candi[t] = setup.models[i].llik(
                        setup.ys[i] - discrep_curr[i][t],
                        pred_curr[i][t],
                        marg_lik_cov_candi[t],
                        wt_mat[i],
                    )

                llik_diffi = llik_candi - llik_curr[i]
                alpha_s2 = setup.itl * (llik_diffi)
                alpha_s2 += (
                    setup.itl
                    * setup.s2_prior_kern[i](
                        np.exp(ls2_candi), setup.ig_a[i], setup.ig_b[i]
                    ).sum(axis=1)
                )  # ldhc_kern(np.exp(ls2_cand[i])).sum(axis=1)#ldig_kern(np.exp(ls2_cand[i]),setup.ig_a[i],setup.ig_b[i]).sum(axis=1)
                alpha_s2 += setup.itl * ls2_candi.sum(axis=1)
                alpha_s2 -= (
                    setup.itl
                    * setup.s2_prior_kern[i](
                        np.exp(log_s2[i][m - 1]), setup.ig_a[i], setup.ig_b[i]
                    ).sum(axis=1)
                )  # ldhc_kern(np.exp(log_s2[i][m-1])).sum(axis=1)#ldig_kern(np.exp(log_s2[i][m-1]),setup.ig_a[i],setup.ig_b[i]).sum(axis=1)
                alpha_s2 -= setup.itl * log_s2[i][m - 1].sum(axis=1)

                runif = np.log(uniform(size=setup.ntemps))
                for t in np.where(runif < alpha_s2)[0]:
                    count_s2[i, t] += 1
                    llik_curr[i, t] = llik_candi[t].copy()
                    log_s2[i][m][t] = ls2_candi[t].copy()
                    marg_lik_cov_curr[i][t] = marg_lik_cov_candi[t].copy()
                    cov_ls2_cand[i].count_100[t] += 1

                cov_ls2_cand[i].update_tau(m)

        #######################
        ### Tempering Swaps ###
        #######################
        if m > setup.start_temper and setup.ntemps > 1:
            for _ in range(setup.nswap):
                sw = np.random.choice(
                    setup.ntemps, 2 * setup.nswap_per, replace=False
                ).reshape(-1, 2)
                sw_alpha[:] = 0.0  # Log Probability of Swap
                sw_alpha += (setup.itl[sw.T[1]] - setup.itl[sw.T[0]]) * (
                    llik_curr[:, sw.T[0]].sum(axis=0)
                    - llik_curr[:, sw.T[1]].sum(axis=0)
                )
                for i in range(setup.nexp):
                    sw_alpha += (setup.itl[sw.T[1]] - setup.itl[sw.T[0]]) * (
                        setup.s2_prior_kern[i](
                            np.exp(log_s2[i][m][sw.T[0]]),
                            setup.ig_a[i],
                            setup.ig_b[i],
                        ).sum(axis=1)
                        - setup.s2_prior_kern[i](
                            np.exp(log_s2[i][m][sw.T[1]]),
                            setup.ig_a[i],
                            setup.ig_b[i],
                        ).sum(axis=1)
                    )
                    if setup.models[i].nd > 0:
                        sw_alpha += (
                            setup.itl[sw.T[1]] - setup.itl[sw.T[0]]
                        ) * (
                            -0.5
                            * (discrep_vars[i][m][sw.T[0]] ** 2).sum(axis=1)
                            / setup.models[i].discrep_tau
                            + 0.5
                            * (discrep_vars[i][m][sw.T[1]] ** 2).sum(axis=1)
                            / setup.models[i].discrep_tau
                        )
                for tt in sw[
                    np.where(np.log(uniform(size=setup.nswap_per)) < sw_alpha)[
                        0
                    ]
                ]:
                    for i in range(setup.nexp):
                        log_s2[i][m][tt[0]], log_s2[i][m][tt[1]] = (
                            log_s2[i][m][tt[1]].copy(),
                            log_s2[i][m][tt[0]].copy(),
                        )
                        (
                            marg_lik_cov_curr[i][tt[0]],
                            marg_lik_cov_curr[i][tt[1]],
                        ) = (
                            marg_lik_cov_curr[i][tt[1]].copy(),
                            marg_lik_cov_curr[i][tt[0]].copy(),
                        )
                        pred_curr[i][tt[0]], pred_curr[i][tt[1]] = (
                            pred_curr[i][tt[1]].copy(),
                            pred_curr[i][tt[0]].copy(),
                        )
                        discrep_curr[i][tt[0]], discrep_curr[i][tt[1]] = (
                            discrep_curr[i][tt[1]].copy(),
                            discrep_curr[i][tt[0]].copy(),
                        )
                        discrep_vars[i][m][tt[0]], discrep_vars[i][m][tt[1]] = (
                            discrep_vars[i][m][tt[1]].copy(),
                            discrep_vars[i][m][tt[0]].copy(),
                        )
                        llik_curr[i, tt[0]], llik_curr[i, tt[1]] = (
                            llik_curr[i, tt[1]].copy(),
                            llik_curr[i, tt[0]].copy(),
                        )
                        # if np.any(np.exp(log_s2[i][m][0]) > 10*np.exp(log_s2[i][m-1][0])):
                        #    print('bummer2')
                    count[tt[0], tt[1]] += 1
                    theta[m][tt[0]], theta[m][tt[1]] = (
                        theta[m][tt[1]].copy(),
                        theta[m][tt[0]].copy(),
                    )

        llik[m] = llik_curr[:, 0].sum()
        # print('\rCalibration MCMC {:.01%} Complete'.format(m / setup.nmcmc), end='')

    s2 = log_s2.copy()
    for i in range(setup.nexp):
        s2[i] = np.exp(log_s2[i])

    theta_native = tran_unif(theta[:, 0], setup.bounds_mat, setup.bounds.keys())

    t1 = time.time()
    print(f"\rCalibration MCMC Complete. Time: {t1 - t0:f} seconds.")
    count = count + count.T - np.diag(np.diag(count))
    out = OutCalibPool(
        theta,
        s2,
        count,
        count_s2,
        count_decor,
        cov_theta_cand,
        cov_ls2_cand,
        pred_curr,
        discrep_vars,
        llik,
        theta_native,
    )
    return out


##############################################################################################################################################################################


class PoolCalib:
    # adapted from https://stackoverflow.com/questions/1816958/cant-pickle-type-instancemethod-when-using-multiprocessing-pool-map/41959862#41959862 answer by parisjohn
    # somewhat slow collection of results
    def __init__(self, setup_list):
        self.setup_list = setup_list

    def singleCal(self, i):
        return calibPool(self.setup_list[i])

    def fit(self, ncores, num):
        pool = Pool(ncores)
        out = pool.map(self, range(num))
        return out

    def __call__(self, i):
        return self.singleCal(i)


def calibPoolParallel(setup_list, ncores):
    temp = PoolCalib(setup_list)
    out = temp.fit(ncores, len(setup_list))
    return out
