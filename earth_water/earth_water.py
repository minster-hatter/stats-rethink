from configparser import ConfigParser

from pymc import Model, Uniform, Binomial, sample
from arviz import summary, plot_trace, plot_posterior

# from numpy import median
from matplotlib.pyplot import savefig

# Constants for later use.
config = ConfigParser()
config.read("../config.ini")
SAMPLES = config.getint("parameters", "SAMPLES")
CHAINS = config.getint("parameters", "CHAINS")
CI = config.getfloat("parameters", "CREDIBLE_INTERVAL")
W_OBSERVED = 6
N = 9

with Model() as m:
    """W ~ Binomial(N, p)
    p ~ Uniform(0, 1)
    """
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
