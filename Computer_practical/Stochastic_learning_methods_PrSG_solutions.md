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

# Stochastic gradient descent preparation {-}  

The function **learning_rate(t,t0)**   computes the learning rate sequence 
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


The function **grad_loss_fun(w,z)**   returns the gradient of the loss function at parameter value $w$, and at example value $z=(x,y)$.   


```r
grad_loss_fun <- function(w,z) {
  x = z[1]
  y = z[2]
  h <- prediction_rule(x,w)
  grd <- c(h-y, (h-y)*x)
  return (grd)
}
```

The function **grad_risk_fun <- function(w,z,n)**   returns the gradient of the risk function at parameter value $w$, and using the data set $z$ of size $n\times 2$.    


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


Computation of the gradient of the empirical risk function at point $w=(-0.1,1.5)^\top$.  

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

 


# Projection Stochastic Gradient Descent with projection  

Let the data set $\mathcal{S}_{n}$ has size $n=1000000$.  

The data set containing the examples to train the model are generated below, and stores in the array $z_{\text{obs}}$.  

Assume that the real values for the unknown parameters $w$ is $w_{\text{true}}=(0.0,1.0)^\top$.  

Assume that the hypothesis class is restricted such as $\mathcal{H}=\{w\in\mathbb{R}^{2}:|w|_2 \le 2.0\} $

You may use the function 


```r
set.seed(2023)
n_obs <- 1000000
w_true <- c(0,1)  
z_obs <- data_generating_model(n = n_obs, w = w_true)
w_true <- as.numeric(glm(z_obs[,2]~ 1+ z_obs[,1],family = "binomial" )$coefficients)
```


## Task 


Compute $w^{*}$ from the minimization problem 

$$
w^{*}=\arg\min_{w\in\mathcal{H}}\left( F(w) \right)
$$
where $F(w)=\left\Vert w-w' \right\Vert$, $w'=(-0.1,0.3)^\top$ and  $\mathcal{H}=\{w\in\mathbb{R}^{2}:|w|_2 \le 2.0\} $. 

Use the function 'nloptr{nloptr}' from the R package 'nloptr'. Try ?nloptr for more information.  

A set of sufficient arguments for the function to run is given below.


```r
# out <- nloptr(x0=..,   
#               eval_f=..., #   
#               eval_grad_f=...,  
#               eval_g_ineq = ...,  
#               eval_jac_g_ineq = ...,   
#               w_now=...,  
#               opts = list("algorithm" = "NLOPT_LD_MMA",  
#                           "xtol_rel"=1.0e-8)  
# out$solution
```


The following functions are provided.


```r
#
boundary <- 2.0 # this is the value |w|_{2}^{2} <= boundary
# auxiliary functions to compute the projection
eval_f0 <- function( w_proj, w_now ){ 
  return( sqrt(sum((w_proj-w_now)^2)) )
}
eval_grad_f0 <- function( w, w_now ){ 
  return( c( 2*(w[1]-w_now[1]), 2*(w[2]-w_now[2]) ) )
}
eval_g0 <- function( w_proj, w_now) {
  return( sum(w_proj^2) -(boundary)^2 )
}
eval_jac_g0 <- function( x, w_now ) {
  return(   c(2*w[1],2*w[2] )  )
}
```

Write your code below


```r
#
w <- c(-0.1,0.3)
#
out <- nloptr(x0=c(0.0,0.0),
            eval_f=eval_f0,
            eval_grad_f=eval_grad_f0,
            eval_g_ineq = eval_g0,
            eval_jac_g_ineq = eval_jac_g0, 
            w_now=w,
            opts = list("algorithm" = "NLOPT_LD_MMA",
                        "xtol_rel"=1.0e-8) 
            )
out$solution
```

```
## [1] -0.1  0.3
```


## Task    

Code a batch Stochastic Gradient Descent (SGD) algorithm with learning rate $\eta_{t}=0.1$, batch size $m=1$ and projection to $\mathcal{H}=\{w:|w|_2\le 2.0\}$, that returns the chain of $\{w^{(t)}\}$. 

The batch sampling may be performed as a sampling with replacement (see ?sample.int). 

The termination criterion is when the total number of iterations excesses $T=1000$. Seed with $w^{(0)}=(-0.3,0.3)^\top$.   

You may use the function 'nloptr{nloptr}' from the R package 'nloptr'. Try ?nloptr for more information.



```r
#
boundary <- 2.0 # this is the value |w|_{2}^{2} <= boundary
# auxiliary functions to compute the projection
eval_f0 <- function( w_proj, w_now ){ 
  return( sqrt(sum((w_proj-w_now)^2)) )
}
eval_grad_f0 <- function( w, w_now ){ 
  return( c( 2*(w[1]-w_now[1]), 2*(w[2]-w_now[2]) ) )
}
eval_g0 <- function( w_proj, w_now) {
  return( sum(w_proj^2) -(boundary)^2 )
}
eval_jac_g0 <- function( x, w_now ) {
  return(   c(2*w[1],2*w[2] )  )
}
```



```r
m <- 1
eta <- 0.1
Tmax <- 1000
w_seed <- c(-0.3,3.0)

w <- w_seed
w_chain <- c(w_seed)
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
  } else {
    zbatch <- z_obs[J,]
  }
  #eta <- learning_rate( t )
  erf_fun <- function(w, z = zbatch, n=m) {
    return( empirical_risk_fun(w, z, n) ) 
  }
  w <- w - eta * numDeriv::grad( erf_fun, w )
  #w <- w - eta * grad_risk_fun( w, zbatch, m )
  # step 1.5 projection
  out <- nloptr(x0=c(0.0,0.0),
              eval_f=eval_f0,
              eval_grad_f=eval_grad_f0,
              eval_g_ineq = eval_g0,
              eval_jac_g_ineq = eval_jac_g0, 
              w_now=w,
              opts = list("algorithm" = "NLOPT_LD_MMA",
                          "xtol_rel"=1.0e-8),
  )
  w <- out$solution
  # record
  w_chain <- rbind(w_chain, w)
  # step 2: check for rtermination terminate
  if ( t>= Tmax ) {
    Qstop <- 1
  }
}
```

```
## 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 990 991 992 993 994 995 996 997 998 999 1000
```

```r
plot(w_chain[,1], type='l') +
abline(h=w_true[1], col='red')
```

![plot of chunk unnamed-chunk-19](figure/unnamed-chunk-19-1.png)

```
## integer(0)
```

```r
plot(w_chain[,2], type='l') +
abline(h=w_true[2], col='red')
```

![plot of chunk unnamed-chunk-19](figure/unnamed-chunk-19-2.png)

```
## integer(0)
```




 
