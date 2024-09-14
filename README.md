## Evolution Strategies for Constraint Optimization Problems  
The main branch of this repository contains all sources for experimentation with the Matrix Adaptation Evolution Strategy in constrained environments.  
  
As a **first step** you need a suitable Python environment.  
Anaconda users may create an admissible environment provided by the yml-file 'es4cop_env'.  
Just execute  
`conda env create -f es4cop_env.yml`  
and  
`conda activate es4cop`  
(<emph>Note: Basically, only Pyhton3.7+, Numpy, and Matplotlib.Pyplot are reqquired.</emph>)
<br><br>
The resources for the experiments are provided in the py-files `maes.py` and `text-functions.py`.
- `text-functions.py` contains the COP formulations of the form
      $\begin{equation}
        \begin{split}
           \min \:\: & f(\mathbf{y})  & & \qquad \gets \quad \textrm{objective function}\\
    s.t. \:\: & g_i(\mathbf{y}) \leq 0,  & i=1,\dots,l,\: & \qquad\gets \quad \textrm{inequality constraints}\\
     & h_j(\mathbf{y}) = 0,     & j=1,\dots,k, & \qquad\gets \quad \textrm{equality constraints}\\
     & \mathbf{y} \in S = \left[\check{\mathbf{y}},\hat{\mathbf{y}}\right]\subseteq \mathbb{R}^N. & & \qquad\gets \quad \textrm{box constraints}
  \end{split}
\end{equation} $

- `maes.py` contains the two variants of the MA-ES and various constraint handling routines
