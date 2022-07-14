from configparser import ConfigParser
from math import e, pi

from pymc import (
    Model,
    Uniform,
    Binomial,
    sample,
    sample_prior_predictive,
    sample_posterior_predictive,
)
from arviz import summary, plot_trace, plot_posterior, plot_ppc
from matplotlib.pyplot import savefig, subplots
from scipy.constants import golden as phi

# Constants for later use.
config = ConfigParser()
config.read("../config.ini")
SAMPLES = config.getint("parameters", "SAMPLES")
CHAINS = config.getint("parameters", "CHAINS")
CI = config.getfloat("parameters", "CREDIBLE_INTERVAL")
W_OBSERVED = 6
N = 9

with Model() as model:
    # W ~ Binomial(N, p)
    # p ~ Uniform(0, 1)
    # Prior.
    p = Uniform("p", 0.0, 1.0)
    # Likelihood.
    W = Binomial("W", N, p, observed=W_OBSERVED)
    # Sample.
    idata = sample(SAMPLES, chains=CHAINS)

summary(idata, hdi_prob=CI, stat_focus="median").to_csv("m_summary.csv")

plot_trace(idata, var_names=["p"], compact=True)
savefig("p_traceplot.png")

plot_posterior(idata, var_names=["p"], point_estimate="median", hdi_prob=CI)
savefig("p_posterior.png")

# Model checking.
with model:
    idata.extend(sample_prior_predictive())
    idata.extend(sample_posterior_predictive(idata))


fig, ax = subplots(2, 1, figsize=(e * pi * phi, e * pi))
plot_ppc(
    idata,
    group="prior",
    observed=False,
    colors=["orange", "black", "darkorange"],
    kind="scatter",
    ax=ax[0],
)
plot_ppc(
    idata, colors=["orange", "black", "darkorange"], kind="scatter", ax=ax[1]
)
savefig("predictive_checks.png")
