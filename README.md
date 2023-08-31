# global-activity-score
## Introduction
### GAS method
Global activity scores <sup>[1]</sup> are introduced as a more accurate approximation to Sobol' indices in certain cases. It is based on global active subspace (GAS) method<sup>[2]</sup>, which is a generalization of active subspace (AS) method by replacing the matrix C with more "global" information. Theoretical results and experiments show that GAS is better when the test function has uncertainties, or it's discontinuous, and is giving similar results for ordinary cases with AS method.

Consider a square-integrable real-valued function $f(\pmb z)$ with domain $\Omega\subset\pmb R^d$ and finite second-order partial derivatives. Suppose that $\Omega$ is equipped with a probability measure with a cumulative distribution function in the form $\pmb F(\pmb z)=F_1(z_1)\cdot \ldots \cdot F_d(z_d)$, where $F_i$ are marginal distribution functions. 

Define a vector function $D_{\pmb z}f:\Omega\times\Omega\rightarrow\pmb R^d$ as follows: 
$$D_{\pmb z}f(\pmb v,\pmb z)=[D_{\pmb z,1}f(v_{1},\pmb z),...,D_{\pmb z,d}f(v_{d},\pmb z)]^T,$$
where
$$D_{\pmb z,i}f(v_{i},\pmb z)=(f(\pmb v_{\lbrace i\rbrace},\pmb z_{-\lbrace i\rbrace})-f(\pmb z))/(v_{i}-z_{i}).$$

Here $\pmb v_{\lbrace i\rbrace}$ corresponds to the $i$ th input of vector $\pmb v$, and $\pmb z_{-\lbrace i\rbrace}$ is the vector of inputs corresponding to those indices in the complement of ${\lbrace i\rbrace}$. 

Define the $d\times d$ matrix $\pmb C$ by
$$\pmb C=E[E[(D_{\pmb z}f)(D_{\pmb z}f)^T|\pmb z]].$$

Do the eigenvalue decomposition of $\pmb C$ and partition the eigenvalues and eigenvectors into two bloacks, we get the global active subspace method.
$$\pmb C=\pmb U\Lambda \pmb U^T,\Lambda=diag(\lambda_1,...,\lambda_d), \lambda_1\geq...\geq \lambda_d\geq 0.$$

And the global activity score for the $i$th parameter, $1\leq i \leq d$, is defined as:
$$\gamma_i(m)=\sum_{j=1}^{m}{\lambda_ju_{ij}^2}.$$

## Usage
Please refer to https://github.com/RuilongYue/global-active-subspace for function GAS() in global_as.py. After getting u and s from GAS(), global activity scores are calculated by choosing proper $m$. Refer to the three examples in this repo for more details.


## References

[1]. The Global Active Subspace Method. (https://arxiv.org/abs/2304.14142)

[2]. The Global Activity scores. (link to be added)
