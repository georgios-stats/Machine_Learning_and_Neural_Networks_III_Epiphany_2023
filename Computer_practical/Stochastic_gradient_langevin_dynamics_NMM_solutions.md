---
title: "Stochastic gradient Langevin dynamics"
subtitle: "...on a binary clasification problem"
author: "Georgios P. Karagiannis @ MATH3431 Machine Learning and Neural Networks III"
output:
  html_notebook: 
    number_sections: true
  word_document: default
  html_document:
    df_print: paged
    number_sections: true
  pdf_document: default
header-includes: 
  - \usepackage{tikz}
  - \usepackage{pgfplots}
  - \usepackage{amsmath}
---

<!-- -------------------------------------------------------------------------------- -->

<!-- Copyright 2023 Georgios Karagiannis -->

<!-- georgios.karagiannis@durham.ac.uk -->
<!-- Associate Professor -->
<!-- Department of Mathematical Sciences, Durham University, Durham,  UK  -->

<!-- This file is part of Machine Learning and Neural Networks III (MATH3431) -->
<!-- which is the material of the course (MATH3431 Machine Learning and Neural Networks III) -->
<!-- taught by Georgios P. Katagiannis in the Department of Mathematical Sciences   -->
<!-- in the University of Durham  in Michaelmas term in 2019 -->

<!-- Machine_Learning_and_Neural_Networks_III_Epiphany_2023 is free software: you can redistribute it and/or modify -->
<!-- it under the terms of the GNU General Public License as published by -->
<!-- the Free Software Foundation version 3 of the License. -->

<!-- Machine_Learning_and_Neural_Networks_III_Epiphany_2023 is distributed in the hope that it will be useful, -->
<!-- but WITHOUT ANY WARRANTY; without even the implied warranty of -->
<!-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the -->
<!-- GNU General Public License for more details. -->

<!-- You should have received a copy of the GNU General Public License -->
<!-- along with Machine_Learning_and_Neural_Networks_III_Epiphany_2023  If not, see <http://www.gnu.org/licenses/>. -->

<!-- -------------------------------------------------------------------------------- -->



[Back to README](https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023/tree/main/Computer_practical#aim)


```r
rm(list=ls())
```


---

***Aim***

Students will become able to:  

+ practice in R,  

+ implement SGLD algorithm in R  

+ recap logistic regression, Normal mixture model (optionally)  

In the computer room, students will practice on the binary classification problem. 

Students are suggested to practice on the Mixture model example at home where the straightforward implementation of the SGLD is meant to fail.  

---

***Reading material***


+ Lecture notes:  
    + Handouts 4 

+ Reference for *R*:  
    + [Cheat sheet with basic commands](https://www.rstudio.com/wp-content/uploads/2016/10/r-cheat-sheet-3.pdf)   

+ Reference of *rmarkdown* (optional, supplementary material):  
    + [R Markdown cheatsheet](https://www.rstudio.com/wp-content/uploads/2016/03/rmarkdown-cheatsheet-2.0.pdf)  
    + [R Markdown Reference Guide](http://442r58kc8ke1y38f62ssb208-wpengine.netdna-ssl.com/wp-content/uploads/2015/03/rmarkdown-reference.pdf)  
    + [knitr options](https://yihui.name/knitr/options)

+ Reference for *Latex* (optional, supplementary material):  
    + [Latex Cheat Sheet](https://wch.github.io/latexsheet/latexsheet-a4.pdf)  
---

***New software***   

+ R package base functions:    
    + set.seed{base} 
    
+ R package numDeriv functions:    
    + grad{numDeriv}  
    
+ R package mvtnorm functions:    
    + dmvnorm{mvtnorm} 
    


```r
# call libraries
library(numDeriv)
library(mvtnorm)
```

---



```r
# Load R package for printing
library(knitr)
```


```r
# Set a seed of the random number generator
set.seed(2023)
```

# Mixture model  


```r
rm( list = ls() )
```

This task is given as a supplementary material, for your information. You can do it at home for practice.   

The following example is given as a homework practice.   

The problem is non convex, there are two modes (maxima) (true values) in the posterior distribution. 

You will see that you will not be able to visit both of the models (maxima) with one run.  

You may need to perform several runs starting your algorithm from different seeds $w^{(0)}$.  

Consider the Bayesian model below

\[
\begin{cases}
z_{i}|w\overset{\text{ind}}{\sim}p_{1}\text{N}\left(w,\sigma^{2}\right)+\left(1-p_{1}\right)\text{N}\left(\phi-w,\sigma^{2}\right);\,i=1,...,n & \text{ sampling distr. }\\
w\sim\text{N}\left(\mu,s^{2}\right) & \text{ prior }
\end{cases}
\]

with fixed parameters $\phi=20$, $p_{1}=0.5$, $\sigma^{2}=5$, $\mu=0.0$, and $s^{2}=100$


```r
data_generating_model <- function(n,w) {
  z <- rep( NaN, times=n )
  p1 <- 0.5 
  p2 <- 1.0-p1
  w <- 5
  phi <-  20
  sig2 <- 5
  lab <- as.numeric(runif(n_obs)>p1)
  z <- lab*rnorm(n, mean = w, sd = sqrt(sig2)) + (1-lab)*rnorm(n, mean = phi-w, sd = sqrt(sig2))
  return(z)
}
```

By inspecting the sampling distribution $f(z|w)$, you will find out that the true values here are $w=5$ and $w=15$ as both of them satisfy the likelihood.  

Let the dataset $\mathcal{S}_{n}$ has size $n=10^({}$.  

Assume that the real values for the parameter $w$ is $w_{\text{true}}\in\{5,15\}$.  

The dataset containing the examples to train the model are generated below, and stores in the array $z_{\text{obs}}$.  


```r
n_obs <- 10^(6)
w_true <- 5 
set.seed(2023)
z_obs <- data_generating_model(n = n_obs, w = w_true) 
set.seed(0)
hist(z_obs)
```

![plot of chunk unnamed-chunk-7](figure/unnamed-chunk-7-1.png)

## Task    

The PDF of the sampling distribution is  

\[
f\left(z_{i}|w\right)=p_{1}\text{N}\left(z_{i}|w,\sigma^{2}\right)+\left(1-p_{2}\right)\text{N}\left(z_{i}|\phi-w,\sigma^{2}\right)
\]

Code a function `log_sampling_pdf(z, w, p1 = 0.5, phi=20, sig2 = 5)' that returns as a value the PDF of sampling distribution in log scale for a single example $z$.  

You may use the function  dnorm{stats} .


```r
log_sampling_pdf <- function(z, w, p1 = 0.5, phi=20, sig2 = 5) {
  log_sampling_pdf <- p1*dnorm(z, mean = w, sd = sqrt(sig2), log = FALSE)
  log_sampling_pdf <- log_sampling_pdf + (1-p1)*dnorm(z, mean = phi-w, sd = sqrt(sig2), log = FALSE)
  log_sampling_pdf <- log(log_sampling_pdf) ;
  return(log_sampling_pdf)
}
```


## Task   

The PDF of the prior distribution is Normal with mean $\mu=0.0$ and variance $s^{2}=100$. 

\[
f\left(w\right)=\text{N}\left(w|\mu=0.0,s^{2}=100\right)
\]

Code in R a function 'log_prior_pdf(w, mu= 0.0, sig2 = 100 )' for the log PDF of the prior distribution of $w$ with mean default values $0.0$ and variance  default values $1000$.

You may use the function  dnorm{stats} .


```r
log_prior_pdf <- function(w, mu = 0.0, sig2 = 100 ) {
  log_pdf <- dnorm(w, mean = mu, sd = sqrt(sig2), log = TRUE)
  return( log_pdf )
}
```



## Task 

Below is given the learning rate function  learning_rate <- function(t, T_0  , T_1 = , C_0  , s_0   )   

\[
\eta_{t}=
\begin{cases}
C_{0}, & t \le T_{0} \\
\frac{C_{0}}{(t-T_{0})^{\varsigma}}, & T_{0}+1 \le t \le T_{1}  \\
\frac{C_{0}}{(T_{1}-T_{0})^{\varsigma}}, & T_{1}+1 \le t 
\end{cases}
\]

for some constants $C_0$ and $\varsigma\in(0.5,1]$.  


```r
learning_rate <- function(t, T_0  , T_1  , C_0  , s_0   ) {
  if ( t <= T_0 ) {
    eta <- C_0
  } else if ( (T_0+1 <= t) && (t <= T_1 ) ) {
    eta <- C_0 / ( (t-T_0) ^ s_0 )
  } else {
    eta <- C_0 / ( (T_1-T_0) ^ s_0 )
  }
  return(eta)
}
```


## Task     

Compute the gradient of the log pdf of the sampling distribution with respect to $w$ at point $w=4.0$ (at the 1st example; i.e. $z_{1}$).  

Do this by using the function 'grad{numDeriv}' from the R package numDeriv. 

E.g., you can use it as numDeriv::grad( fun, w ). You can try ?grad for more info.


```r
w <- 4.0
#
aux_fun <- function(w, z = z_obs[1]) {
  return( log_sampling_pdf(z, w) ) 
}
#
gr <- numDeriv::grad( aux_fun, w )
#
gr
```

```
## [1] 0.6026276
```



## Task    

Code a Stochastic Gradient Langevin Dynamics (SGLD) algorithm with batch size $m=?$, and temperature $\tau=?$ that returns the chain of all the  $\{w^{(t)}\}$ produced. The learning rate can be constant for the first half of the run, with a decay $\varsigma=?$, and constant $C_{0}=?$.  

The termination criterion is when the total number of iterations excesses $T_{\text{max}}=?$. 

Seed with $w^{(0)}=0$.   

You may use the R function  grad{numDeriv}  to numerically compute the gradient; e.g.  numDeriv::grad( erf_fun, w ) . Try  ?grad  for more info.  

After finishing your code, try to set the algorithmic parameters $m$, $\tau$, $\varsigma$, and constant $C_{0}$ and $T_{\text{max}}$, etc., with purpose to run the code and discover both modes. 

Plot the trace plot and the histogram of the generated chain $\{w^{(t)}\}$. Did you visited both of the areas around the true values of $w$?. You may find out that the produced chain is prone to local trapping aka it is trapped to one mode and unable to cross the valley of low mass in order to visit the other mode. How to do modify GD, SGD, and SGLD in order to address such an issue, remains a challenge.  



```r
Tmax <- 500
#
w_seed <- 0.0
#
eta <- 10^(-2)
eta_C <- eta
eta_s <- 0.51
eta_T0 <- 0.3*Tmax
eta_T1 <- 0.6*Tmax
#
batch_size <- 1000
#
tau <- 1.0
#
# Set the seed
w <- w_seed
w_chain_clipping <- c(w)
# iterate
t <- 1
Qterm <- 0
#
clipping_threshold <- 10
#
# iterate
#
while ( (Qterm != 1) ) {
  # counter 
  t <- t+1
  cat( t ) ; cat( ' ' ) ## counter added for display reasons
  # learning rate
  eta <- learning_rate(t, eta_T0, eta_T1, eta_C, eta_s)
  # sub-sample
  J <- sample.int( n = n_obs, size = batch_size, replace = FALSE)
  # update
  ## likelihood
  grad_est_lik <- rep( 0.0, times=length(w) )
  for (j in J) {
    aux_fun <- function(w, z=z_obs[j]){
      gr <- log_sampling_pdf(z, w)
      return(gr)
    }
    grad_est_lik <- grad_est_lik + numDeriv::grad(aux_fun, w)
  }
  grad_est_lik <- ( n_obs / batch_size) * grad_est_lik
  # gradient clipping/rescaring
  norm_grad_est_lik <- sqrt(sum(grad_est_lik^2))
  grad_est_lik <- grad_est_lik * min( 1.0, clipping_threshold/norm_grad_est_lik )
  w <- w +eta*grad_est_lik ; 
  ## prior
  aux_fun <- function(w){
    d <- length(w)
    gr <- log_prior_pdf(w, rep(0,d), 100*diag(d))
    return(gr)
  }
  w <- w +eta*numDeriv::grad(aux_fun, w) ;
  ## noise
  w <- w +sqrt(eta)*sqrt(tau)*rnorm(n = length(w), mean = 0, sd = 1)
  # termination criterion
  if  ( t >= Tmax ) {
    Qterm <- 1
  }
  # record the produced chain
  w_chain_clipping <- rbind(w_chain_clipping,w)
}
```

```
## 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 495 496 497 498 499 500
```


```r
plot(w_chain_clipping, type = 'l')
```

![plot of chunk unnamed-chunk-13](figure/unnamed-chunk-13-1.png)

```r
hist(w_chain_clipping)
```

![plot of chunk unnamed-chunk-13](figure/unnamed-chunk-13-2.png)








