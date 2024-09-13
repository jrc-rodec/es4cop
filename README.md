 
<img src="img/FHVlogo.png" align="right" width="300">
<br>
<h2>Tutorial:</h2>
<h1>Introduction to <br> Evolution Strategies for  Constrained Optimization Problems </h1>
<br>  
<p style="text-align:center;font-size: 22pt">Michael Hellwig, Steffen Finck, and Hans-Georg Beyer</p>
<br>
<p style="text-align:center;font-size: 17pt">Josef Ressel Centre for Robust Decision Making, Vorarlberg University of Applied Sciences, Dornbirn, Austria.</p>

<br>

<p style="text-align:center;font-size: 12pt">The financial support by the Austrian Federal Ministry of Labour and Economy, the National Foundation for Research, Technology and Development and by the Christian Doppler Research Association is gratefully acknowledged. <img src="img/CDGlogo.png" align="right" width="220"></p>
<br>
<br>
<br>

Please find the experimentation sources in the `main` branch of this repository.<br>
<br>

Go to: [https://github.com/jrc-rodec/es4cop](https://github.com/jrc-rodec/es4cop)
<br>

Have fun! <br>
<img src="img/test-combi_t.png" align="center" width="400">
A general continuous single-objective constrained optimization problem (COP)

<br><br>

$$
\begin{equation}
 \begin{aligned}
   \min \:\: & f(\mathbf{y})  & & \qquad \gets \quad \textrm{objective function}\\
    s.t. \:\: & g_i(\mathbf{y}) \leq 0,  & i=1,\dots,l,\: & \qquad\gets \quad \textrm{inequality constraints}\\
     & h_j(\mathbf{y}) = 0,     & j=1,\dots,k, & \qquad\gets \quad \textrm{equality constraints}\\
     & \mathbf{y} \in S = \left[\check{\mathbf{y}},\hat{\mathbf{y}}\right]\subseteq \mathbb{R}^N. & & \qquad\gets \quad \textrm{box constraints}
  \end{aligned}
\end{equation} 
$$

<br><br>

- The continuous (non-)linear objective function is given by $f : \mathbb{R}^N \to \mathbb{R}$.  
- $\mathbf{y} \in \mathbb{R}^N$ represents the $N$-dimensional real-valued parameter vector. 
- The total number of (non-)linear constraint functions is $m=k+l$. 
- The box constraints represent lower ($\check{\mathbf{y}}$) and upper bounds ($\hat{\mathbf{y}}$) on the parameter vector components. 

