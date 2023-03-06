---
title: "Stochastic learning methods"
subtitle: "...on a binary classification problem"
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

+ practice in R,  

+ implement GD, batch/online SGD, AdaGrad, SGD with projection, SVRG algorithms in R

+ refresh logistic regression, with Ridge penalty from term 1  

---

***Reading material***


+ Lecture notes:  
    + Handouts 0, 1, 2, and 3 

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

+ R package `base` functions:    
    + `set.seed{base}` 

+ R package `nloptr` functions:    
    + `nloptr{nloptr}` 
    
+ R package `numDeriv` functions:    
    + `grad{numDeriv}` 


```r
# call libraries
#install.packages("numDeriv")
library(numDeriv)
#install.packages("nloptr")
library(nloptr)
```

---  

***Initialize R***  



```r
# Load R package for printing
library(knitr)
```


```r
# Set a seed of the randon number generator
set.seed(2023)
```


# Application: Binary classification problem {-}

Consider the binary classification problem with input $x\in\mathbb{R}$ and output/labels $y\in\{0,1\}$.  

The prediction rule is 
\[
h_{w}(x) = \frac {\exp(w^\top x)}{1+\exp(w^\top x)}
\] 
where $w\in\mathbb{R}^{2}$ is the unknown parameter we wish to learn, and hence we can consider that the hypothesis class is 

\[
\mathcal{H}=\{w\in\mathbb{R}^{2}\}
\]  

Consider there is available a dataset  
\[
\mathcal{S}_{n}=\left\{ z_{i}=\left(y_{i},x_{i}\right)\right\} _{i=1}^{n}
\]  

with $y_{i}\in\{0,1\}$ and $x_{i}\in\mathbb{R}$.  

Recall that the sampling distribution is  
\[
y|w \sim \text{Bernulli}(h_{w}(x)) \\
h_{w}(x) = \frac {\exp(w^\top x)}{1+\exp(w^\top x)}
\]

The dataset $\mathcal{S}_{n}=\{z_i=(x_i,y_i)\}$ is  generated from the data generation probability $g(\cdot)$ provided below as a routine. We pretend that we do not know $g(\cdot)$. 


```r
data_generating_model <- function(n,w) {
  z <- rep( NaN, times=n*2 )
  z <- matrix(z, nrow = n, ncol = 2)
  z[,1] <- runif(n, min = -10, max = 10)
  p <- w[1] + w[2]*z[,1] 
  p <- exp(p) / (1+exp(p))
  z[,2] <- rbinom(n, size = 1, prob = p)
  return(z)
}
```

Let the dataset $\mathcal{S}_{n}$ has size $n=500$.  

Assume that the real values for the unknown parameters $w$ is $w_{\text{true}}=(0.0,1.0)^\top$.  

The dataset containing the examples to train the model are generated below, and stores in the array $z_{\text{obs}}$.  


```r
set.seed(2023)
n_obs <- 500
w_true <- c(0,1)  
z_obs <- data_generating_model(n = n_obs, w = w_true) 
w_true <- as.numeric(glm(z_obs[,2]~ 1+ z_obs[,1],family = "binomial" )$coefficients)
```

The prediction rule is 
\[
h_{w}(x) = \frac {\exp(w^\top x)}{1+\exp(w^\top x)}
\]  

where $w\in\mathbb{R}^{2}$.

The function **prediction_rule(x,w)** that returns the rule $h$ where $x$ is the input argument and $w$ is the unknown parameter is given below.  


```r
prediction_rule <- function(x,w) {
  h <- w[1]+w[2]*x
  h <- exp(h) / (1.0 + exp(h) )
  return (h)
}
```

The likelihood function is 
$$
f\left(y|w\right)=\prod_{i=1}^{n}\left(h_{i}\right)^{y_{i}}\left(1-h_{i}\right)^{1-y_{i}}
$$

We consider a loss function as   

\[
\ell\left(w,z=\left(x,y\right)\right)=-y\log\left(h(x)\right)-\left(1-y\right)\log\left(1-h(x)\right)
\]

The code for the loss function is provided below as **loss_fun(w,z)** that computes the loss function, where $z=(x,y)$ is one example (observation) and $w$ is the unknown parameter. 


```r
loss_fun <- function(w,z) {
  x = z[1]
  y = z[2]
  h <- prediction_rule(x,w)
  ell <- -y*log(h) - (1-y)*log(1-h)
  return (ell)
}
```

The Risk function under the data generation model $g$ is 

\[
\begin{align*}
R_{g}\left(w\right)= & \text{E}\left(\ell\left(w,z=\left(x,y\right)\right)\right)\\
= & \text{E}\left(-y\log\left(h\left(w;x\right)\right)-\left(1-y\right)\log\left(1-h\left(w;x\right)\right)\right)
\end{align*}
\]

The Empirical risk function is
\[
\begin{align*}
\hat{R}_{S}\left(w\right) & \frac{1}{n}\sum_{i=1}^{n}\ell\left(w,z_{i}=\left(x_{i},y_{i}\right)\right)\\
= & -\frac{1}{n}\sum_{i=1}^{n}\left(y_{i}\log\left(h(w;x_{i})\right)+\left(1-y\right)\log\left(1-h(w;x_{i})\right)\right)
\end{align*}
\]

The function **empirical_risk_fun(w,z,n)** computes the empirical risk, where $z=(x,y)$ is an example, $w$ is the unknown parameter, and $n$ is the data size is given below. 


```r
empirical_risk_fun <- function(w,z,n) {
  x = z[,1]
  y = z[,2]
  R <- 0.0
  for (i in 1:n) {
    R <- R + loss_fun(w,z[i,])
  }
  R <- R / n
  return (R)
}
```

# Preparation for the (Stochastic) Gradient Descent {-}  

Code a function **learning_rate(t,t0)** that computes the learning rate sequence 
\[
\eta_{t}=\frac{t_0}{t}
\]  
where $t$ is the iteration stage and $t_0$ is a constant. 

Use $t_0=3$ as default value.  


```r
learning_rate <-function(t,t0=3) {
  eta <- t0 / t
  return( eta )
}
```


## Task (given) 

Code the function **grad_loss_fun(w,z)** that returns the gradient of the loss function at parameter value $w$, and at example value $z=(x,y)$.   


```r
grad_loss_fun <- function(w,z) {
  x = z[1]
  y = z[2]
  h <- prediction_rule(x,w)
  grd <- c(h-y, (h-y)*x)
  return (grd)
}
```

## Task (given) 

Code the function **grad_risk_fun <- function(w,z,n)** that returns the gradient of the risk function at parameter value $w$, and using the data set $z$ of size $n\times 2$.    


```r
grad_risk_fun <- function(w,z,n) {
  grd <- 0.0
  for (i in 1:n) {
    grd <- grd + grad_loss_fun(w,z[i,])
  }
  grd <- grd / n
  return (grd)
}
```


## Task (do it in the computer practical)    

Compute the gradient of the empirical risk function at point $w=(-0.1,1.5)^\top$.  

Use the whole dataset $\{z_{i};i=1,...,n\}$ (set of examples). 

Do this by using the command 'grad_risk_fun' provided above.


```r
w <- c(-0.1,1.5)
gr <- grad_risk_fun (w, z=z_obs, n=n_obs) 
gr
```

```
## [1] -0.005427957  0.045414831
```

## Task (do it in the computer practical)    

Compute the gradient of the empirical risk function at point $w=(-0.3,3)^\top$. Use the whole dataset $\{z_{i};i=1,...,n\}$ (set of examples).   Do this by using the function 'grad{numDeriv}' from the R package numDeriv. 

E.g., you can use it as numDeriv::grad( fun, w ). You can try ?grad for more info.


```r
w <- c(-0.1,1.5)
#
erf_fun <- function(w, z = z_obs, n=n_obs) {
  return( empirical_risk_fun(w, z, n) ) 
}
#
gr <- numDeriv::grad( erf_fun, w )
#
gr
```

```
## [1] -0.005427957  0.045414831
```

# Gradient descent  

## Task (do it in the computer practical)    

Code a Gradient Descent (GD) algorithm with constant learning rate $\eta_{t}=0.5$ that returns the chain of all the  $\{w^{(t)}\}$ produced. 

The termination criterion is such that the iterations stop when the the total number of iterations excesses $T=300$. 

Use seed $w^{(0)}=(-0.3,3)^\top$.   

You may use the R function **grad{numDeriv}** to numerically compute the gradient;  

+ e.g. numDeriv::grad( erf_fun, w ) .  

+ Try ?grad for more info. 


```r
eta <- 0.5
Tmax <- 300
w_seed <- c(-0.3,3)
w <- w_seed
w_chain <- c()
Qstop <- 0 
t <- 0
while ( Qstop == 0 ) {
  # counter
  t <- t +  1
  cat( t ) ; cat( ' ' ) ## counter added for display reasons
  # step 1: update  
  erf_fun <- function(w, z = z_obs, n=n_obs) {
    return( empirical_risk_fun(w, z, n) ) 
  }
  w <- w - eta * numDeriv::grad( erf_fun, w )
  w_chain <- rbind(w_chain, w)
  # step 2: check for termination terminate
  if ( t>= Tmax ) {
    Qstop <- 1
  }
}
```

```
## 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300
```



```r
# eta <- 0.1
# Tmax <- 1000
# w_seed <- c(-0.1,1.5)
# w <- w_seed
# w_chain <- c()
# Qstop <- 0 
# t <- 0
# while ( Qstop == 0 ) {
#   # counter
#   t <- t +  1
#   #eta <- learning_rate( t )
#   # step 1: update 
#   w <- w - eta * grad_risk_fun( w, z_obs, n_obs )
#   w_chain <- rbind(w_chain, w)
#   # step 2: check for termination terminate
#   if ( t>= Tmax ) {
#     Qstop <- 1
#   }
# }
```

## Task (do it in the computer practical)    

Plot the chain $\{w_1^{(t)}\}$ against the iteration $t$.   


```r
plot(w_chain[,1], type='l') +
abline(h=w_true[1], col='red')
```

![plot of chunk unnamed-chunk-17](figure/unnamed-chunk-17-1.png)

```
## integer(0)
```

Plot the chain $\{w_2^{(t)}\}$ against the iteration $t$.  


```r
plot(w_chain[,2], type='l') +
abline(h=w_true[2], col='red')
```

![plot of chunk unnamed-chunk-18](figure/unnamed-chunk-18-1.png)

```
## integer(0)
```

## Task (do it in the computer practical)    

Re-run the previous GD by changing the algorithminc parameter values for $\eta$ for some in the range $(0.001,1.0)$.  

Check how the algorithm behaves by ploting the chains $\{w_1^{(t)}\}$ and $\{w_2^{(t)}\}$ against the iteration $t$.    

If necessary change the termination criterion to consider more or less iterations.  


```r
#
#
#
```

## Task (for homework practice)   

Re run GD by using a learning rate sequence of the form $\eta_t = t_0/t$ for different values of $t_0>0$ that you will choose.  

Check how the algorithm behaves by plotting the chains $\{w_1^{(t)}\}$ and $\{w_2^{(t)}\}$ against the iteration $t$.  

If necessary change the termination criterion to consider more or less iterations.


```r
learning_rate <- function(t,t0) {
  return(t0/t)
}
t0<- 10
Tmax <- 300
w_seed <- c(-0.3,3.0)
w <- w_seed
w_chain <- c()
Qstop <- 0 
t <- 0
while ( Qstop == 0 ) {
  # counter
  t <- t +  1
  cat( t ) ; cat( ' ' ) ## counter added for display reasons
  # step 1: update  
  eta <- learning_rate( t, t0 )
  erf_fun <- function(w, z = z_obs, n=n_obs) {
    return( empirical_risk_fun(w, z, n) ) 
  }
  w <- w - eta * numDeriv::grad( erf_fun, w )
  #w <- w - eta * grad_risk_fun( w, z_obs, n_obs )
  w_chain <- rbind(w_chain, w)
  # step 2: check for rtermination terminate
  if ( t>= Tmax ) {
    Qstop <- 1
  }
}
```

```
## 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300
```

```r
plot(w_chain[,1], type='l') +
abline(h=w_true[1], col='red')
```

![plot of chunk unnamed-chunk-20](figure/unnamed-chunk-20-1.png)

```
## integer(0)
```

```r
plot(w_chain[,2], type='l') +
abline(h=w_true[2], col='red')
```

![plot of chunk unnamed-chunk-20](figure/unnamed-chunk-20-2.png)

```
## integer(0)
```

# Batch Stochastic Gradient Descent 

Let the data set $\mathcal{S}_{n}$ has size $n=1000000$.  

Assume that the real values for the unknown parameters $w$ is $w_{\text{true}}=(0.0,1.0)^\top$.  

The dataset containing the examples to train the model are generated below, and stored in the array $z_{\text{obs}}$.  


```r
set.seed(2023)
n_obs <- 1000000
w_true <- c(0,1)  
z_obs <- data_generating_model(n = n_obs, w = w_true)
w_true <- as.numeric(glm(z_obs[,2]~ 1+ z_obs[,1],family = "binomial" )$coefficients)
```

## Task (do it in the computer practical)    

Code a batch Stochastic Gradient Descent (GD) algorithm with learning rate $\eta_{t}=0.5$ and batch size $m=10$ that returns the chain of $\{w^{(t)}\}$.  

The batch sampling may be performed as a sampling with replacement (see ?sample.int).  

The termination criterion is when the total number of iterations excesses $T=300$. Seed with $w^{(0)}=(-0.3,3)^\top$.   


```r
m <- 10
eta <- 0.5
Tmax <- 300
w_seed <- c(-0.3,3)
w <- w_seed
w_chain <- c()
Qstop <- 0 
t <- 0
while ( Qstop == 0 ) {
  # counter
  t <- t +  1
  cat( t ) ; cat( ' ' ) ## counter added for display reasons
  # step 1: update  
  J <- sample.int(n = n_obs, size = m, replace = TRUE)
  if (m==1) {
    zbatch <- matrix(z_obs[J,],1,2)
  }
  else {
    zbatch <- z_obs[J,]
  }
  #eta <- learning_rate( t )
  erf_fun <- function(w, z = zbatch, n=m) {
    return( empirical_risk_fun(w, z, n) ) 
  }
  w <- w - eta * numDeriv::grad( erf_fun, w )
  #w <- w - eta * grad_risk_fun( w, zbatch, m )
  w_chain <- rbind(w_chain, w)
  # step 2: check for rtermination terminate
  if ( t>= Tmax ) {
    Qstop <- 1
  }
}
```

```
## 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300
```

```r
plot(w_chain[,1], type='l') +
abline(h=w_true[1], col='red')
```

![plot of chunk unnamed-chunk-22](figure/unnamed-chunk-22-1.png)

```
## integer(0)
```

```r
plot(w_chain[,2], type='l') +
abline(h=w_true[2], col='red')
```

![plot of chunk unnamed-chunk-22](figure/unnamed-chunk-22-2.png)

```
## integer(0)
```


## Task (do it in the computer practical)    

Re run the batch SGD by experimenting and changing the values of the learning rate $\eta$ and that of the batch size $m$.

Plot the produced chains of $\{w^{(t)}\}$. 

What is the impact of the the learning rate $eta$ and that of the batch size $m$ to the noise and the speed of the convergence ?  



```r
m <- 80
eta <- 0.5
Tmax <- 300
w_seed <- c(-0.3,3)
w <- w_seed
w_chain <- c()
Qstop <- 0 
t <- 0
while ( Qstop == 0 ) {
  # counter
  t <- t +  1
  cat( t ) ; cat( ' ' ) ## counter added for display reasons
  # step 1: update  
  J <- sample.int(n = n_obs, size = m, replace = TRUE)
  if (m==1) {
    zbatch <- matrix(z_obs[J,],1,2)
  }
  else {
    zbatch <- z_obs[J,]
  }
  #eta <- learning_rate( t )
  erf_fun <- function(w, z = zbatch, n=m) {
    return( empirical_risk_fun(w, z, n) ) 
  }
  w <- w - eta * numDeriv::grad( erf_fun, w )
  #w <- w - eta * grad_risk_fun( w, zbatch, m )
  w_chain <- rbind(w_chain, w)
  # step 2: check for rtermination terminate
  if ( t>= Tmax ) {
    Qstop <- 1
  }
}
```

```
## 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300
```

```r
#
plot(w_chain[,1], type='l') +
abline(h=w_true[1], col='red')
```

![plot of chunk unnamed-chunk-23](figure/unnamed-chunk-23-1.png)

```
## integer(0)
```

```r
#
plot(w_chain[,2], type='l') +
abline(h=w_true[2], col='red')
```

![plot of chunk unnamed-chunk-23](figure/unnamed-chunk-23-2.png)

```
## integer(0)
```

```r
# 
```
ANSWER; As discussed in the lectures, the bigger the batch size the smaller the variation of the gradient, hence the error is smaller.  


## Additional tasks  

### AdaGrad  

What would you do it you wish the learning rate to be automatically adjusted?    

Practice on the following variation.  

+ [LINK TO TASKS](http://htmlpreview.github.io/?https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023/tree/main/Computer_practical/Stochastic_learning_methods_AdaGrad_tasks.nb.html)  

+ [LINK TO SOLUTIONS](http://htmlpreview.github.io/?https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023/tree/main/Computer_practical/Stochastic_learning_methods_AdaGrad_solutions.nb.html)  

### Projection  

What would you do if the parametric space / hypothesis class is constrained?  

Practice on the following variation.  

+ [LINK TO TASKS](http://htmlpreview.github.io/?https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023/tree/main/Computer_practical/Stochastic_learning_methods_PrSG_tasks.nb.html)  

+ [LINK TO SOLUTIONS](http://htmlpreview.github.io/?https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023/tree/main/Computer_practical/Stochastic_learning_methods_PrSG_solutions.nb.html)  

### Variance reduction  

What would you do it you wanted to reduce the variance of the stochastic gradient?  

Practice on the following variation.  

+ [LINK TO TASKS](http://htmlpreview.github.io/?https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023/tree/main/Computer_practical/Stochastic_learning_methods_VRSG_tasks.nb.html)  

+ [LINK TO SOLUTIONS](http://htmlpreview.github.io/?https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023/tree/main/Computer_practical/Stochastic_learning_methods_VRSG_solutions.nb.html) 



