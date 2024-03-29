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
# Set a seed of the randon number generator
set.seed(2023)
```


# Application: Binary classification problem  {-}

Consider the binary classification problem with input $x\in\mathbb{R}$ and output/labels $y\in\{0,1\}$. 

Consider there is available a data set  

$$
\mathcal{S}_{n}=\left\{ z_{i}=\left(y_{i},x_{i}\right)\right\} _{i=1}^{n}
$$

with $y_{i}\in\{0,1\}$ and $x_{i}\in\mathbb{R}$.  

The dataset $\mathcal{S}_{n}$ is  generated from the data generation probability $g(\cdot)$ provided below as a routine. We pretend that we do not know $g(\cdot)$. 


```r
data_generating_model <- function(n,w) {
  d <- 3
  z <- rep( NaN, times=n*d )
  z <- matrix(z, nrow = n, ncol = d)
  z[,1] <- 1.0
  z[,2] <- runif(n, min = -10, max = 10)
  p <- w[1]*z[,1] + w[2]*z[,2] 
  p <- exp(p) / (1+exp(p))
  z[,3] <- rbinom(n, size = 1, prob = p)
  return(z)
}
```

Let the dataset $\mathcal{S}_{n}$ has size $n=10^{6}$.  

Assume that the real values for the unknown parameters $w$ is $w_{\text{true}}=(0.0,1.0)^\top$.  

The dataset containing the examples to train the model are generated below, and stores in the array $z_{\text{obs}}$.  


```r
n_obs <- 10^(6)
w_true <- c(0,1)  
set.seed(2023)
z_obs <- data_generating_model(n = n_obs, w = w_true) 
set.seed(0)
w_true <- as.numeric(glm(z_obs[,3]~ 1 + z_obs[,2],family = "binomial" )$coefficients)
```


The predictive rule $h_{w}\left(x\right)$ is 

\[
h_{w}\left(x\right)=\frac{\exp\left(x^{\top}w\right)}{1+\exp\left(x^{\top}w\right)}
\]

where $w\in\mathbb{R}^{2}$ is the unknown parameter we wish to learn. The hypothesis class is $$\mathcal{H}=\{w\in\mathbb{R}^{2}\}$$.   

Write a function `prediction_rule(x,w)' that returns the rule $h$ where $x$ is the input argument and $w$ is the unknown parameter.


```r
prediction_rule <- function(x,w) {
  h <- w[1]*x[1]+w[2]*x[2]
  h <- exp(h) / (1.0 + exp(h) )
  return (h)
}
```

The Bayesian model is 

\[
\begin{cases}
y_{i}|w\sim\text{Bernoulli}\left(h_{w}\left(x\right)\right),\,\,i=1,...,n & \text{( sampling distribution)}\\
\,\,\,\,\,\,\,\,\,\,h_{w}\left(x\right)=\frac{\exp\left(x^{\top}w\right)}{1+\exp\left(x^{\top}w\right)}\\
w\sim\text{N}_{d}\left(\mu=0.0,V=I_{d}\right) & \text{( prior distribution)}
\end{cases}
\]

We consider $d=2$, $V=100 I_{2}$, and $\mu=0.0$.

The log PDF of the sampling distribution is 

\[
\log\left(f\left(y|w\right)\right)=y\log\left(h_{w}\left(x\right)\right)+\left(1-y\right)\log\left(1-h_{w}\left(x\right)\right)
\]

Here is coded the R function 'log_sampling_pdf(z, w)' for the log PDF of the sampling distribution


```r
log_sampling_pdf <- function(z, w) {
  d <- length(w)
  x <- z[1:d] 
  y <- z[d+1]
  log_pdf <- y * log(prediction_rule(x,w)) +(1-y) * log( 1.0-prediction_rule(x,w) )
  #log_pdf <- dbinom(y, size = 1, prob = prediction_rule(x,w), log = TRUE)
  return( log_pdf )
}
```

The log PDF of the prior distribution is 

\[
\log\left(f\left(w\right)\right)=-\frac{d}{2}\log\left(2\pi\right)-\frac{1}{2}\left|V\right|-\frac{1}{2}\left(\beta-\mu\right)^{\top}V^{-1}\left(\beta-\mu\right)
\]

Here is coded tha R function **log_prior_pdf(w, mu= rep(0, length(w)), Sig2 = 1000*diag(length(w)) )** for the log PDF of the prior distribution of $w$ with mean default values $0.0$ and variance matrix default values $V=\text{diag}(1000,...,1000)$. 

You may use the R function **dmvnorm{mvtnorm}**


```r
log_prior_pdf <- function(w, mu, Sig2 ) {
  log_pdf <- dmvnorm(w, mean = mu, sigma = Sig2, log = TRUE, checkSymmetry = TRUE)
  return( log_pdf )
}
```


# Stochastic Gradient Langevin Dynamics (SGLD)

Consider the learning rate function   

\[
\eta_{t}=
\begin{cases}
C_{0}, & t \le T_{0} \\
\frac{C_{0}}{(t-T_{0})^{\varsigma}}, & T_{0}+1 \le t \le T_{1}  \\
\frac{C_{0}}{(T_{1}-T_{0})^{\varsigma}}, & T_{1}+1 \le t 
\end{cases}
\]

for some constants $C_0$ and $\varsigma\in(0.5,1]$.  

The R coded function **learning_rate <- function(t, T_0 = 100, T_1 = 500, C_0 = 0.0001, s_0 = 0.5 )**  is given below   


```r
learning_rate <- function(t, T_0 = 100, T_1 = 500, C_0 = 0.0001, s_0 = 0.5 ) {
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


## Task   (for the computer practical)   

Code a Stochastic Gradient Langevin Dynamics (SGLD) algorithm with batch size $m=0.1 n$, and temperature $\tau=1.0$ that returns the chain of all the  $\{w^{(t)}\}$ produced. 

Consider a learning rate which is constant around $C_{0}=10^{-6}$ for the $1/3$ of the run, then it can delay $\varsigma=0.51$, and constant  $C_{0}=10^{-6}$, while in the final $1/3$ it can be constant to a small value. (use the aforesaid provided function for the learning rate).  

The termination criterion is when the total number of iterations excesses $T_{\text{max}}= 500$. 

Seed with $w^{(0)}=(-1,0)^\top$.   

You may use the R function **grad{numDeriv}** to numerically compute the gradient; e.g. **numDeriv::grad( erf_fun, w )**. Try **?grad** for more info.  



```r
Tmax <- 500
#
w_seed <- c(-1,0)
#
eta <- 10^(-6)
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
w_chain <- c(w)
# iterate
t <- 1
Qterm <- 0
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
    aux_fun <- function(w, z=z_obs[j,]){
      gr <- log_sampling_pdf(z, w)
      return(gr)
    }
    grad_est_lik <- grad_est_lik + numDeriv::grad(aux_fun, w)
  }
  grad_est_lik <- ( n_obs / batch_size) * grad_est_lik
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
  w_chain <- rbind(w_chain,w)
}
```

```
## 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 495 496 497 498 499 500
```


## Task (for the computer practical)   

Plot the trace plots of chains $\{w_1^{(t)}\}$ and $\{w_2^{(t)}\}$ against the iteration $t$.  

If you are not happy with the convergence, feel free to tune the algorithmic parameters in the previous code properly, and run it again.  


```r
plot(w_chain[,1], type='l') +
abline(h=w_true[1], col='red')
```

![plot of chunk unnamed-chunk-12](figure/unnamed-chunk-12-1.png)

```
## integer(0)
```

```r
#
plot(w_chain[,2], type='l') +
abline(h=w_true[2], col='red')
```

![plot of chunk unnamed-chunk-12](figure/unnamed-chunk-12-2.png)

```
## integer(0)
```

## Task (for the computer practical)   

Code a Stochastic Gradient Langevin Dynamics (SGLD) algorithm with batch size $m=0.1n$, and temperature $\tau=1.0$ that returns the chain of all the  $\{w^{(t)}\}$ produced.  

The learning rate can be constant around $C_{0}=10^{-2}$ for the $1/3$ of the run, then it can delay $\varsigma=0.51$ with constant $C_{0}=10^{-2}$, while in the final $1/3$ it can be constant to a small value. (use the aforesaid provided function for the learning rate). 

The termination criterion is when the total number of iterations excesses $T_{\text{max}}=500$. 

Seed with $w^{(0)}=(-1,0)^\top$.   

You may use the R function 'grad{numDeriv}' to numerically compute the gradient; e.g. numDeriv::grad( erf_fun, w ) . Try ?grad for more info.  

If the gradient explodes apply gradient clipping/scaling with threshold $C=10.0$, as 

\[
w^{(t+1)}=w^{(t)}+\eta_{t}\left(\text{clip}\left(\frac{n}{m}\sum_{j\in\mathcal{J}^{(t)}}\nabla_{w}\log\left(f\left(z^{(t)}|w^{(t)}\right)\right),c\right)+\nabla_{w}\log\left(f\left(w^{(t)}\right)\right)\right)+\sqrt{\eta_{t}\tau}\epsilon_{t},\,\epsilon_{t}\sim\text{N}\left(0,1\right)
\]
 where 
\[
\text{clip}\left(v,c\right)=v\min\left(1,\frac{c}{\left\Vert v\right\Vert }\right)
\]
 and $v$ is the gradient. 


```r
Tmax <- 500
#
w_seed <- c(-1,0)
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
    aux_fun <- function(w, z=z_obs[j,]){
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
plot(w_chain[,1], type='l') +
abline(h=w_true[1], col='red')
```

![plot of chunk unnamed-chunk-14](figure/unnamed-chunk-14-1.png)

```
## integer(0)
```

```r
#
plot(w_chain[,2], type='l') +
abline(h=w_true[2], col='red')
```

![plot of chunk unnamed-chunk-14](figure/unnamed-chunk-14-2.png)

```
## integer(0)
```


## Task (for the computer practical)   

Let's go back to the code without using the clipping gradient.  

Based on the above, copy the tail end of the generated chain $w^(t)$ after discarding the burn in, as 'w_chain_output'.  


```r
T_bunrin <- as.integer(0.6*Tmax)
w_chain_output <- w_chain[T_bunrin:Tmax,]
```



## Task (for the computer practical)   

Plot the histograms plots of output chains $\{w_1^{(t)}\}$ and $\{w_2^{(t)}\}$ for the estimation of the marginal posterior distributions of the dimensions of $w$. 


```r
hist(w_chain_output[,1]) 
```

![plot of chunk unnamed-chunk-16](figure/unnamed-chunk-16-1.png)

```r
#
hist(w_chain_output[,2]) 
```

![plot of chunk unnamed-chunk-16](figure/unnamed-chunk-16-2.png)

## Task  

To learn $\text{E}_{f}\left(w_{1}+w_{2}|y\right)$, compute the estimator 

\[
\widehat{w_{1}+w_{2}}=\frac{1}{T}\sum_{t=1}^{T}\left(w_{1}^{(t)}+w_{2}^{(t)}\right)
\]

based on SGLD output chain.


```r
w_est = mean(rowMeans(w_chain_output))
w_est
```

```
## [1] 0.5049435
```



## Task   

Use the prediction rule  

\[
h_{w}\left(x\right)=\frac{\exp\left(x^{\top}w\right)}{1+\exp\left(x^{\top}w\right)}
\]

and the estimates for $w$ in order to classify the example with feature $x_\text{new}=c(1, 0.5)$.  

Particularly, compute the point estimate  

\[
\widehat{h_{w}\left(x\right)}=\frac{1}{T}\sum_{t=1}^{T}h_{w^{(t)}}\left(x\right)=\frac{1}{T}\sum_{t=1}^{T}\frac{\exp\left(x^{\top}w^{(t)}\right)}{1+\exp\left(x^{\top}w^{(t)}\right)}
\]



```r
x_new <- c(1,0.5)
T <- dim(w_chain_output)[1]
h_est <- 0.0 
for (t in 1:T) {
  h_est <- h_est + prediction_rule( x_new , w_chain_output[t,] )
}
h_est <- h_est / T
h_est
```

```
## [1] 0.6249346
```

Now, compute the point estimate  

\[
\widehat{h_{w}\left(x\right)}=h_{\hat{w}}\left(x\right)=\frac{\exp\left(x^{\top}\hat{w}\right)}{1+\exp\left(x^{\top}\hat{w}\right)}
\]

by just plugin in the estimate $\hat{w}$. 


```r
x_new <- c(1,0.5)
w_est <- colMeans(w_chain_output)
h_est <- prediction_rule( x_new , w_est )
h_est
```

```
## [1] 0.6249354
```

Now, estimate the pdf of the prediction rule at the example with feature $x_\text{new}=c(1, 0.5)$ (to represent the uncertainty) by using a histogram  


```r
x_new <- c(1,0.5)
T <- dim(w_chain_output)[1]
h_chain <- c()
for (t in 1:T) {
  h_chain <- c( h_chain , 
                prediction_rule( x_new , w_chain_output[t,] ) 
                )
}
hist(h_chain)
```

![plot of chunk unnamed-chunk-20](figure/unnamed-chunk-20-1.png)


## Additional tasks  

### Normal mixture model  

This task is given as a supplementary material, for your information and your practice.  

What happens when the learning problem is non convex? Aka where there are multiple modes (maxima) (true values) in the posterior distribution?   

You will see that you will not be able to visit both of the models (maxima) with one run.  

+ [LINK TO TASKS](http://htmlpreview.github.io/?https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023/tree/main/Computer_practical/Stochastic_gradient_langevin_dynamics_NMM_tasks.nb.html)  

+ [LINK TO SOLUTIONS](http://htmlpreview.github.io/?https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023/tree/main/Computer_practical/Stochastic_gradient_langevin_dynamics_NMM_solutions.nb.html)  









