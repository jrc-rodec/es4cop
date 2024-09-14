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
- `text-functions.py` contains the COP formulations 
- `maes.py` contains the two variants of the MA-ES and various constraint handling routines
