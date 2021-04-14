############################################################################################
#
# This code implements the Self-Learning Algorithm (SLA*) described in:
#
#   Amini, Massih, Nicolas Usunier, and François Laviolette.
#   "A transductive bound for the voted classifier
#   with an application to semi-supervised learning."
#   In Advances in Neural Information Processing Systems, pp. 65-72. 2009.
#
# Version: 0.1 (March, 2018)
#
#
# Copyright (C) 2018 Vasilii Feofanov
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
# INSTALLATION and DEPENDENCIES
# 1. Python 3
# 2. Numpy
# 3. Scikit-Learn
# The code was tested on Python 3.6.8 with Numpy 1.16.4, Scikit-Learn 0.21.2
#
############################################################################################

from sklearn.ensemble import RandomForestClassifier
import numpy as np


def compute_joint_bayes_risk(margins, risk_gibbs, theta, sampling_rate=50):
    u = len(margins)
    gammas = theta + (1-theta)*(np.arange(sampling_rate)+1)/sampling_rate
    infimum = 1e+05
    upper_bounds = []
    for n in range(np.size(gammas)):
        gamma = gammas[n]
        prob_between = np.sum((margins < gamma) & (margins > theta))/u
        K = risk_gibbs + 0.5*(np.sum(margins)/u-1)
        # M-less of gamma
        Mg = np.sum(margins[margins < gamma])/u
        # M-no-greater of theta
        Mt = np.sum(margins[margins <= theta])/u
        A = K + Mt - Mg
        upper_bound = prob_between + (A*(A > 0))/gamma
        upper_bounds.append(upper_bound)
        if upper_bound < infimum:
            infimum = upper_bound
        if n > 3:
            if upper_bounds[-1] > upper_bounds[-2] and upper_bounds[-2] >= upper_bounds[-3]:
                break
    return infimum


def optimal_threshold(x_u, margins, risk_gibbs, sampling_rate = 50):
    u = x_u.shape[0]
    # A set of possible thetas
    theta_min = np.min(margins)
    theta_max = np.max(margins)
    thetas = theta_min + np.arange(sampling_rate) * (theta_max - theta_min) / sampling_rate

    def compute_cond_bayes_err_one_theta(i):
        theta = thetas[i]
        joint_bayes_risk = compute_joint_bayes_risk(margins, risk_gibbs, theta)
        prob_be_labeled = np.sum(margins > theta)/u
        if prob_be_labeled == 0:
            prob_be_labeled = 1e-15
        conditional_bayes_err = joint_bayes_risk / prob_be_labeled
        return conditional_bayes_err

    compute_cond_bayes_err = np.vectorize(compute_cond_bayes_err_one_theta)
    bayes_err = compute_cond_bayes_err(np.arange(len(thetas)))
    min_idx = np.argmin(bayes_err)
    if type(min_idx) is list:
        min_idx = min_idx[0]
    theta_star = thetas[min_idx]

    return theta_star


def SLA(x_l, y_l, x_u, **kwargs):
    """
    A margin-based self-learning algorithm.
    :param x_l: Labeled observations.
    :param y_l: Labels.
    :param x_u:  Unlabeled data. Will be used for learning.
    :return: The final classification model H that has been trained on the labeled
    and the pseudo-labeled unlabeled examples.
    """

    if 'n_estimators' not in kwargs:
        n_est = 200
    else:
        n_est = kwargs['n_estimators']

    if 'random_state' not in kwargs:
        random_state = np.random.randint(low=0, high=1000, size=1)[0]
    else:
        random_state = kwargs['random_state']

    classifier = RandomForestClassifier(n_estimators=n_est, oob_score=True, random_state=random_state)
    l = x_l.shape[0]
    sample_distr = np.repeat(1/l, l)
    b = True
    thetas = []
    while b:
        # Learn a classifier
        H = classifier
        H.fit(x_l, y_l, sample_weight=sample_distr)
        # Margin estimation
        probs = H.predict_proba(x_u)
        margins = abs(probs[:, 1] - probs[:, 0])
        labels = np.sign(probs[:, 1] - probs[:, 0])

        # An upper bound of the Gibbs risk is set to 0.5
        risk_gibbs = 0.5

        # Find a threshold minimizing Bayes conditional error
        theta = optimal_threshold(x_u, margins, risk_gibbs)
        thetas.append(theta)

        # Select observations with margins more than theta
        x_s = x_u[margins >= theta, :]
        y_s = labels[margins >= theta]
        # Stop if there is no anything to add:
        if x_s.shape[0] == 0:
            b = False
            continue
        # Move them from the test set to the train one
        x_l = np.concatenate((x_l, x_s))
        y_l = np.concatenate((y_l, y_s))
        x_u = np.delete(x_u, np.where(margins >= theta), axis=0)
        s = x_l.shape[0]-l
        sample_distr = np.concatenate((np.repeat(1/l, l), np.repeat(1/s, s)))

        # Stop criterion
        if x_u.shape[0] == 0:
            b = False
    return H, thetas
