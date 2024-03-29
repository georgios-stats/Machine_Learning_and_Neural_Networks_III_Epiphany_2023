---
title: "Artificial Neural Networks"
subtitle: "Introductory analysis via NNET"
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

+ implement Feed Forward Neural Network with R package nnet in R.   

---

***Reading material***


+ Lecture notes:  
    + [Handout 5: Artificial neural networks](https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023/blob/main/Lecture_handouts/05.Artificial_neural_networks.pdf)  
    + [Ripley, B., Venables, W., & Ripley, M. B. (2016). Package ‘nnet’. R package version, 7(3-12), 700.](https://cran.r-project.org/web/packages/nnet/nnet.pdf)  

+ Reference for *R*:  
    + [Cheat sheet with basic commands](https://www.rstudio.com/wp-content/uploads/2016/10/r-cheat-sheet-3.pdf)   

+ Reference of *rmarkdown* (optional given as supplementary material):  
    + [R Markdown cheatsheet](https://www.rstudio.com/wp-content/uploads/2016/03/rmarkdown-cheatsheet-2.0.pdf)  
    + [R Markdown Reference Guide](http://442r58kc8ke1y38f62ssb208-wpengine.netdna-ssl.com/wp-content/uploads/2015/03/rmarkdown-reference.pdf)  
    + [knitr options](https://yihui.name/knitr/options)

+ Reference for *Latex* (optional given as supplementary material):  
    + [Latex Cheat Sheet](https://wch.github.io/latexsheet/latexsheet-a4.pdf)  

---

***New software***   

+ R package `base` functions:    
    + `set.seed{base}` 

+ R package `nnet` functions:    
    + `nnet{nnet}` , `class.ind{nnet}` , `predict{nnet}` , `which.is.max{nnet}` 

---



```r
# Load R package for printing
library(knitr)
```


```r
# Set a seed of the randon number generator
set.seed(2023)
```

# Familiarity with analysis with feed-forward Neural Networks with nnet

We will use the R package nnet. It is available from  

+ [https://cran.r-project.org/web/packages/nnet/](https://cran.r-project.org/web/packages/nnet/)  

The reference manual is available from  

+ [https://cran.r-project.org/web/packages/nnet/nnet.pdf](https://cran.r-project.org/web/packages/nnet/nnet.pdf)  

Details

+ nnet fits single-hidden-layer neural network, possibly with skip-layer connections.  

## Task: Install nnnet (given)  


```r
## build version (recommended)
#install.packages("nnet")
## linux build
#install.packages("https://cran.r-project.org/src/contrib/nnet_7.3-18.tar.gz", repos = NULL, type = "source")
## windows build
#install.packages("https://cran.r-project.org/bin/windows/contrib/4.3/nnet_7.3-18.zip", repos = NULL, type = "source")
```
## Task: Load the R package nnet (given)  


```r
library(nnet)
```

## Task: About nnet commands (to be done at home)  

Check out R package nnet commands from the reference manual is available from  

+ [https://cran.r-project.org/web/packages/nnet/nnet.pdf](https://cran.r-project.org/web/packages/nnet/nnet.pdf) 

in particular:  nnet{nnet}, predict.nnet{nnet}, which.is.max{nnet}, and class.ind{nnet}.

# Regression problem with 1 output  

This is the "Case 1. (Regression problem)" from the [Handout 5: Artificial neural networks](https://raw.githubusercontent.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023/main/Lecture_handouts/05.Artificial_neural_networks.pdf).    

## Task: Ozone data set (given)  

We will use the OZON data from the textbook  

+ Faraway, J. J. (2016). Extending the linear model with R: generalized linear, mixed effects and nonparametric regression models. Chapman and Hall/CRC.  

A study the relationship between atmospheric ozone concentration and meteorology in the Los Angeles Basin in 1976. A number of cases with missing variables have been removed for simplicity.  

This is a data frame with 330 observations on the following 10 variables.

Install and load the R package "faraway".  


```r
## build version (recommended)
#install.packages("faraway")
## linux build
#install.packages("https://cran.r-project.org/src/contrib/faraway_1.0.8.tar.gz", repos = NULL, type = "source")
## windows build
#install.packages("https://cran.r-project.org/bin/windows/contrib/4.3/faraway_1.0.8.zip", repos = NULL, type = "source")
library("faraway")
```

Load the data set ozone{faraway}


```r
data(ozone)
```

Read the description by "?ozone".  


```r
?ozone
```

Print "ozone" data set.   


```r
ozone
```

```
##     O3   vh wind humidity temp  ibh dpg ibt vis doy
## 1    3 5710    4       28   40 2693 -25  87 250  33
## 2    5 5700    3       37   45  590 -24 128 100  34
## 3    5 5760    3       51   54 1450  25 139  60  35
## 4    6 5720    4       69   35 1568  15 121  60  36
## 5    4 5790    6       19   45 2631 -33 123 100  37
## 6    4 5790    3       25   55  554 -28 182 250  38
## 7    6 5700    3       73   41 2083  23 114 120  39
## 8    7 5700    3       59   44 2654  -2  91 120  40
## 9    4 5770    8       27   54 5000 -19  92 120  41
## 10   6 5720    3       44   51  111   9 173 150  42
## 11   5 5760    6       33   51  492 -44 181  40  43
## 12   4 5780    6       19   54 5000 -44 135 200  44
## 13   4 5830    3       19   58 1249 -53 243 250  45
## 14   7 5870    2       19   61 5000 -67 186 200  46
## 15   5 5840    5       19   64 5000 -40 174 200  47
## 16   9 5780    4       59   67  639   1 189 150  48
## 17   4 5680    5       73   52  393 -68 210  10  49
## 18   3 5720    4       19   54 5000 -66 126 140  50
## 19   4 5760    3       19   54 5000 -58 111 250  51
## 20   4 5730    4       26   58 5000 -26 111 200  52
## 21   5 5700    5       59   69 3044  18 116 150  53
## 22   6 5650    5       70   51 3641  23  87 140  54
## 23   9 5680    3       64   53  111 -10 153  50  55
## 24   6 5820    5       19   59  597 -52 214  70  57
## 25   6 5810    5       19   64 1791 -15 182 150  59
## 26  11 5790    3       28   63  793 -15 188 120  60
## 27  10 5800    2       32   63  531 -38 244  40  61
## 28   7 5820    5       19   62  419 -29 243 120  61
## 29  12 5770    8       76   63  816  -7 190   6  62
## 30   9 5670    3       69   54 3651  62  95  30  63
## 31   2 5590    3       76   36 5000  70  33 100  64
## 32   3 5410    6       64   31 5000  28   2 200  65
## 33   3 5350    7       62   30 1341  18  77  60  66
## 34   2 5480    9       72   36 5000   0  37 350  67
## 35   3 5600    7       76   42 3799 -18  77 250  68
## 36   3 5490   11       72   37 5000  32  34 350  69
## 37   4 5560   10       72   41 5000  -1  31 300  70
## 38   6 5700    3       32   46 5000 -30  77 300  71
## 39   8 5680    5       50   51 5000  -8  75 300  72
## 40   6 5700    4       86   55 2398  21 121 200  73
## 41   4 5650    5       61   41 5000  51  24 100  74
## 42   3 5610    5       62   41 4281  42  52 250  75
## 43   7 5730    5       66   49 1161  27 116 200  76
## 44  11 5770    5       68   45 2778   2 132 200  77
## 45  13 5770    3       82   55  442  26 146  40  78
## 46   6 5690    8       21   41 5000 -30  57 300  80
## 47   5 5700    3       19   45 5000 -53  66 300  81
## 48   4 5730   11       19   51 5000 -43  95 300  82
## 49   4 5690    7       19   53 5000   7  95 300  83
## 50   6 5640    5       68   50 5000  24  56 300  84
## 51  10 5720    6       63   60 1341  19 151 150  85
## 52  15 5740    3       54   54 1318   2 181 150  86
## 53  23 5740    3       47   53  885  -4 195  80  87
## 54  17 5740    3       56   53  360   3 195  40  88
## 55   7 5670    7       61   44 3497  73  97  40  89
## 56   2 5550   10       74   40 5000  73  45  80  91
## 57   3 5470    7       46   30 5000  44 -15 300  92
## 58   3 5320   11       45   25 5000  39 -25 200  93
## 59   4 5530    3       43   40 5000 -12   9 140  95
## 60   6 5600    3       21   45 5000  -2  39 140  96
## 61   7 5660    7       57   51 5000  30  56 140  97
## 62   7 5580    5       42   48 3608  24  41 100  98
## 63   6 5510    5       50   45 5000  38   5 140  99
## 64   3 5530    5       61   47 5000  56  20 200 100
## 65   2 5620    9       61   43 5000  66  13 120 101
## 66   8 5690    0       60   49  613 -27 154 300 102
## 67  12 5760    4       31   56  334  -9 180 300 103
## 68  12 5740    3       66   53  567  13 166 150 104
## 69  16 5780    5       53   61  488 -20 183   2 105
## 70   9 5790    2       42   63  531 -15 217  50 106
## 71  24 5760    3       60   70  508   7 192  70 107
## 72  13 5700    4       82   57 1571  68 135  17 108
## 73   8 5680    4       57   35  721  28 130 140 109
## 74  10 5720    5       21   52  505 -49 196 140 110
## 75   8 5720    5       19   59  377 -27 229 300 111
## 76   9 5730    4       32   67  442  -9 243 200 112
## 77  10 5710    5       77   57  902  54 158 250 113
## 78  14 5720    4       71   42 1381   4 135  60 115
## 79   9 5710    3       19   55 5000 -16 100 100 116
## 80  11 5600    6       45   40 5000  38  83 150 117
## 81   7 5630    4       44   39 1302  40 115 150 118
## 82   9 5690    7       70   57 1292  -5 120 200 119
## 83  12 5730    6       45   58 5000 -14 115 100 120
## 84  12 5710    3       46   62  472  34 172 300 121
## 85   8 5610    6       50   51 1404  42 125 120 121
## 86   9 5680    5       69   61  944  35 132 100 122
## 87   5 5620    6       67   34 5000  75  18 200 123
## 88   4 5420    7       69   35 5000  41  -6 200 124
## 89   4 5540    5       54   35 5000  62   8 200 125
## 90   9 5590    6       51   48 5000  44  56 300 126
## 91  13 5690    6       63   59 2014  31 119 300 127
## 92   5 5550    7       63   41 5000  56  29 250 128
## 93  10 5620    7       57   58 5000  27  87 120 129
## 94  10 5630    6       61   51  524  57 126 140 130
## 95   7 5580    7       78   46 5000  55  36 200 131
## 96   5 5560    4       65   40 5000  59  18 140 132
## 97   4 5440    5       44   35 5000  24   3  80 133
## 98   7 5480    7       51   46 2490  29  86 300 134
## 99   3 5620    5       73   39 5000 107  -4 100 135
## 100  4 5450   11       35   32 5000  36   8 300 136
## 101  7 5660    6       35   47 5000  28  41 200 137
## 102 11 5680    6       61   50 1144  30 120 120 138
## 103 15 5760    4       50   65  547   1 194 100 139
## 104 22 5790    4       57   66  413  10 209 120 140
## 105 17 5720    5       68   69  610  46 176  60 141
## 106  7 5660    6       58   59 3638  81 107 120 142
## 107 10 5710    5       65   64 3848  45 138 100 143
## 108 19 5780    7       78   68 1479  40 200 100 144
## 109 18 5750    7       73   49 1108  55 186  27 145
## 110 12 5700    5       41   52  869   0 145  40 146
## 111  6 5620    9       47   56 5000  43  35 140 147
## 112  9 5650    6       46   55 5000  49  33 150 148
## 113 19 5730    5       61   66 1148  31 160 100 149
## 114 21 5810    4       55   74  856   4 241 100 150
## 115 29 5790    4       60   76  807  16 228 120 151
## 116 16 5740    8       78   70 2040  46 175 150 152
## 117 11 5690    4       71   67  314  60 150 120 154
## 118  2 5680    6       77   41 5000  75  49 120 156
## 119 12 5650    8       66   61 1410  20 129 140 157
## 120 16 5730    6       74   68  360  23 169 120 158
## 121 22 5730    3       78   69 1568  32 198  70 159
## 122 20 5760    7       78   74 1184  40 204  80 160
## 123 27 5830    6       75   74  898  24 230  70 161
## 124 33 5880    3       80   80  436   0 302  40 162
## 125 25 5890    6       88   84  774   6 300  20 163
## 126 31 5850    4       76   78 1181  50 266  17 164
## 127 18 5820    6       63   80 1991  47 209  40 165
## 128 24 5800    7       78   76 1597  56 200  50 167
## 129 16 5740    3       74   74 1184  52 208  70 168
## 130 12 5710    7       63   66 3005  58 151  80 169
## 131  9 5720    8       62   66 2880  53 141 120 170
## 132 16 5740    5       53   69 2125  64 150 100 172
## 133  8 5690    9       62   62 3720  74 105 120 174
## 134  9 5730    5       71   67 4337  66 152 200 175
## 135 29 5780    3       68   80 2053  31 227 120 176
## 136 20 5790    7       79   76 1958  70 214  40 177
## 137  5 5750    3       76   65 3644  86 152  70 178
## 138  5 5680    6       71   65 1368  75 147 100 179
## 139 11 5720    3       66   63 3539  73 120 120 180
## 140 12 5770    4       81   62 2785  49 174 100 181
## 141 19 5800    4       72   68  984  26 207 120 181
## 142 17 5780    8       92   68 1804  56 200  70 182
## 143 19 5740    6       71   69 3234  77 171  80 183
## 144 16 5730    6       64   66 3441  67 161 100 184
## 145 14 5760    6       68   70 1578  61 160 100 185
## 146 10 5770    7       59   70 1850  76 160 120 186
## 147  9 5690    8       67   64 2962  80 152 120 187
## 148  7 5650    6       66   61 2670  54 130 120 188
## 149  5 5610    3       61   52 5000  76  56 150 189
## 150  2 5570    9       81   48 5000  57  49 140 190
## 151 12 5690    5       63   59 5000  46 107 140 191
## 152 22 5760    3       58   67  987  28 177 140 192
## 153 17 5810    5       68   66 1148  43 194 140 193
## 154 26 5830    4       71   74  898 -24 255  60 194
## 155 27 5880    6       67   83  777  -1 281  30 195
## 156 14 5860    3       64   78 1279  75 220  17 196
## 157 11 5830    6       64   75 1046  69 204  80 197
## 158 23 5870    4       69   84 1167  50 235  60 198
## 159 26 5860    3       77   81  987  45 243 100 199
## 160 21 5800    3       61   79 1144  57 218 120 200
## 161 15 5800    4       69   79  977  60 215 150 201
## 162 20 5770    5       64   65  770  26 242 120 202
## 163 15 5860    4       33   81  629 -11 302 140 203
## 164 18 5870    7       38   84  337 -14 321 140 204
## 165 26 5870    4       54   83  590  26 295 120 205
## 166 19 5860    6       39   90  400  19 285 120 206
## 167 13 5880    5       43   90  580   9 307  80 207
## 168 30 5870    7       55   93  646  25 318 140 208
## 169 26 5860    4       77   88  826  41 291 140 209
## 170 15 5830    5       63   72  823  52 236 150 210
## 171 16 5820    5       65   72 2116  47 213 120 211
## 172 16 5820    8       64   70 2972  52 180 120 212
## 173 19 5860    6       68   78 2752  41 211 140 213
## 174 23 5870    3       76   87 1377  37 258 100 214
## 175 28 5890    6       71   91 1486  33 266  50 215
## 176 34 5900    6       86   87  990  22 295  40 216
## 177 33 5890    5       65   91  508  29 296 100 217
## 178 24 5910    4       73   88 1204  56 266 100 219
## 179 17 5900    5       69   83 2414  63 247  60 220
## 180 10 5860    3       64   78 2385  67 213  50 221
## 181 14 5830    3       63   79 2326  64 218  70 222
## 182 13 5850    9       72   77 3389  56 204  80 223
## 183 17 5830    6       82   81 2818  58 221  80 224
## 184 22 5810    8       69   76 2394  54 209  90 226
## 185 19 5830    4       74   78 2746  61 208 120 227
## 186 20 5830    5       69   75 2493  55 225 120 228
## 187 25 5840    7       72   82 1528  42 233 100 229
## 188 28 5870    6       73   84  111  40 256  60 230
## 189 29 5870    4       90   86 1899  45 247  40 231
## 190 23 5860    3       80   80 1289  32 240  40 233
## 191 26 5900    3       73   80  984  35 260  70 234
## 192 14 5890    4       71   84  836  28 275  80 235
## 193 13 5880    4       78   84  826  27 263  80 236
## 194 26 5890    6       80   81 1105  39 234  80 237
## 195 22 5870    8       74   85 1023  46 251  80 238
## 196 14 5820    6       63   73 2956  46 196 120 241
## 197 13 5780    6       57   72 2988  56 187 150 241
## 198  9 5770    3       55   68 4291  60 168 200 242
## 199 12 5790    4       65   65 3330  59 148 150 243
## 200 14 5840    7       65   79 1233  30 214 100 249
## 201 24 5910    5       72   81 1069  28 235  80 251
## 202 19 5890    5       79   80  984  57 230  70 252
## 203 16 5870    6       62   76 1653  71 204  60 253
## 204  7 5780    7       65   59 3930  68 151 150 254
## 205  2 5730    5       77   55 5000  73 109 200 255
## 206  4 5780    7       70   66 5000  45 107 200 256
## 207  6 5750    7       58   64 4212  46 138 200 257
## 208 12 5760    5       58   62 5000  52  99 250 258
## 209  9 5730    7       72   67 5000  31 141 300 259
## 210 15 5730    5       77   74 1545  43 187  70 260
## 211 17 5790    4       57   74  994  44 209 300 261
## 212 13 5750    3       67   70 1125  55 200 150 262
## 213 20 5880    3       73   77  636  16 233 300 263
## 214 22 5890    7       70   83  748  32 250  30 264
## 215 24 5880    4       73   81  692  44 254 100 265
## 216 26 5870    7       73   73  807  39 260 100 266
## 217 32 5900    6       71   87  869  19 261  17 267
## 218 33 5920    4       77   89  800  24 298  20 268
## 219 27 5930    3       68   92  393   6 332   4 269
## 220 38 5950    5       62   92  557   0 326  70 270
## 221 23 5950    8       61   93  620  27 298  30 271
## 222 19 5900    5       71   93 1404  33 293  70 271
## 223 19 5890    8       77   86  898  21 270  60 272
## 224 15 5860    7       71   76  377  -2 285  40 273
## 225 28 5840    5       67   81  528  17 260  50 274
## 226 10 5800    6       74   78 2818  26 226  70 275
## 227 14 5760    7       65   73 3247  10 196 140 276
## 228 26 5810    6       82   80  895   0 256 100 277
## 229 17 5850    4       67   81  721   0 268 120 278
## 230  3 5760    7       87   52 5000  39 110 150 281
## 231 14 5860    4       71   63 1965  13 161  50 282
## 232 29 5830    5       77   72 1853  10 216  70 283
## 233 18 5840    5       78   75 2342   7 219  40 284
## 234  3 5800    7       72   55 5000  56 109  70 285
## 235  9 5790    3       71   61 4028  35 128 140 287
## 236 19 5830    5       71   71 2716  26 176 140 288
## 237  8 5810    5       76   71 3671  31 188 100 289
## 238 23 5780    6       76   72 3795  31 194  70 291
## 239 13 5800    6       73   75 3120  35 194  40 292
## 240  7 5800    5       80   65 2667  17 175 100 294
## 241  3 5780    9       73   61 5000  39 115 120 295
## 242  5 5790    8       80   60 5000  36  94 120 296
## 243 11 5770    5       75   64  308  25 204 140 297
## 244 12 5750    4       68   61 2982  18 155 120 298
## 245  5 5640    5       93   63 5000  30 115  70 299
## 246  4 5640    7       57   62 5000  25 107 150 300
## 247  5 5650    3       70   59 5000  38  87 200 301
## 248  4 5710    6       65   56 5000  35  88 200 302
## 249 10 5760    6       66   59 3070  13 156 200 303
## 250 17 5840    4       73   72  830   0 223  70 304
## 251 26 5880    3       77   71  711  -9 242  40 305
## 252 30 5890    5       80   75 1049 -10 261  50 306
## 253 18 5890    4       73   71  511 -39 288  17 307
## 254 12 5890    5       19   71 5000 -40 198  80 308
## 255  7 5890    6       19   73 5000 -34 208 250 309
## 256 15 5850    3       73   78  377  -3 260 200 310
## 257 12 5830    5       76   73  862  27 231   2 311
## 258  7 5830    8       77   71  337 -17 273  20 312
## 259 28 5860    5       86   73  492  -2 279   7 313
## 260 22 5830    5       76   71 1394  13 239  30 314
## 261 18 5800    7       66   66 3146  27 178  50 315
## 262 14 5830    4       74   69 2234  11 193  70 316
## 263 24 5790    5       71   69 2109  21 209  17 317
## 264 10 5730    4       84   64 5000  23 125  80 318
## 265 14 5780    5       74   65 2270  -7 205  50 319
## 266  9 5740    7       48   54 2191 -13 204  60 320
## 267 12 5710    8       75   62 3448  12 148  60 321
## 268  7 5690    6       74   56 5000  13  94  80 322
## 269  7 5670    4       67   55 5000  11  97  50 323
## 270  6 5760    4       75   58 2719  25 138  50 324
## 271 13 5820    5       71   48 1899  21 167  40 325
## 272  5 5790    3       35   54 5000 -41 114  40 326
## 273  3 5760    5       23   57 5000 -21 105 300 327
## 274  7 5800    6       19   60 5000 -19 124 200 328
## 275  8 5810    7       59   61 2385  10 158 150 329
## 276 10 5750    4       60   63 1938   0 170 100 330
## 277 12 5840    0       38   65  590 -11 211 100 331
## 278  6 5920    3       22   71  328 -40 270 150 333
## 279  5 5860    7       19   70 5000 -29 165 300 335
## 280 20 5840    0       45   68  597 -22 231  30 337
## 281 14 5810    2       47   69  469  -4 221  50 339
## 282 16 5770    2       73   59 1541  18 173  20 340
## 283  5 5710    4       67   49 5000  24  55 200 341
## 284  3 5500    9       56   39 5000  15  54 120 342
## 285  5 5660    3       54   50 5000  27  70 300 343
## 286  1 5700    3       71   46 5000  54  60 200 344
## 287  5 5810    5       59   54 5000 -28 120  70 345
## 288  4 5860    0       25   60 5000 -38 175 140 346
## 289 11 5900    0       24   62 5000 -36 156 150 347
## 290  6 5850    5       41   65 2014 -20 211 200 348
## 291  8 5780    3       50   66  436   1 213   4 349
## 292 14 5790    0       76   66  830   3 189  40 350
## 293 18 5780    2       82   63 1112  -8 191  30 351
## 294 12 5770    2       81   62 1210 -17 199  30 352
## 295  9 5750    2       85   60  501 -22 216   2 353
## 296  7 5780    5       76   63  875 -15 205   0 354
## 297 14 5790    5       66   60 1601   7 167  30 355
## 298  4 5750    6       58   58 5000  59  55  60 356
## 299  3 5670    8       19   34 5000 -63  28 150 357
## 300  3 5760    0       19   36 5000 -52  50 100 358
## 301  3 5770    4       19   44 2280 -54 132 250 359
## 302  3 5810    2       19   53 2047 -43 175 150 360
## 303  3 5810    2       19   52 5000 -69 136 200 361
## 304  3 5870    3       19   53 3720 -50 163 200 362
## 305  3 5830    2       27   58  311 -24 211 200 363
## 306  6 5760    0       64   55 2536  28 136  80 364
## 307  6 5680    0       52   50 1154 -22 164  60 365
## 308  5 5780    4       19   48 2933 -40 155 300 366
## 309  3 5810    3       19   51 3064 -33 171 200 367
## 310  4 5760    0       32   62  826 -16 182 300 368
## 311  7 5680    0       58   40 5000   2  61  50 369
## 312  5 5750    0       26   44  111 -52 201  40 370
## 313  5 5790    5       19   49 5000 -48 126  70 371
## 314  4 5770    3       19   53 5000 -37 131 150 372
## 315  3 5750    0       19   53 5000 -26 106 150 373
## 316  2 5720    0       19   53 5000 -31 108  70 374
## 317  5 5760    3       19   55  948 -48 215 200 375
## 318  3 5780    0       19   51 5000 -50 105 120 376
## 319  4 5660    4       19   54 5000 -22  92 150 377
## 320  4 5610    2       58   48 3687 -10  83 150 378
## 321  6 5640    0       51   53 5000   0  68  60 379
## 322  6 5680    3       52   49 5000 -19  76  70 380
## 323  3 5650    5       19   48 5000 -28  74 150 381
## 324  4 5710    4       19   51 5000 -25  91 300 382
## 325  3 5680    4       57   47  508 -10 148 100 383
## 326  8 5630    4       50   50 2851  -5 100  70 384
## 327  2 5730    3       53   51  111 -14 225 200 387
## 328  3 5690    3       23   51 5000 -36 107  70 388
## 329  5 5650    3       61   50 3704  18  83  40 389
## 330  1 5550    4       85   39 5000   8  44 100 390
```



## Task: A naive training of NN (to be done in the computer practical class) 

We wish to model as a neural network the predictive rule $h_{w}\left(x\right)$ that  

+ receives as input features $x$ the variables temp, ibh, ibt, from the ozone{faraway} dataset,  

+ and predicts (returns) as output the variable O3 from the ozone{faraway} data set.  

Use the R function nnet{nnet} to fit a feed forward neural network with: 

+ one hidden layer, 

+ $2$ units in the hidden layer.  	

+ a predictive rule as  $h_{w}\left(x\right)=o_{T}\left(x\right)=\sigma_{T}\left(\alpha_{T}\left(x\right)\right)$  

+ the output activation function is linear; i.e. the identity function  $\sigma_{T}\left(a\right)=a$.  

As inputs, we consider the variables temp, ibh, ibt, from the ozone{faraway} dataset.  

R implementation: 

+ Use the command nnet{nnet} with arguments:  

  + formula: stated inputs / outpu,  
  
  + data, 
  
  + size: number of neurons in the hidden layer, 
  
  + linout: TRUE for the regression problem, and FALSE for the classification problem. 



```r
nn.out.1 <- nnet(O3 ~ temp + ibh + ibt, 
                 ozone, 
                 size=2, 
                 linout=TRUE)
```

```
## # weights:  11
## initial  value 69488.006925 
## final  value 21115.406061 
## converged
```
## Task: A naive training of NN (to be done in the computer practical class) 

Let us denote the training dataset as usual by $\left\{ z_{i}=\left(x_{i},y_{i}\right)\right\}$.  

Compute the produced error function from the naively trained NN  

\[
\text{EF}\left(w|z\right)=\sum_{i=1}^{n}\left(h_{w}\left(x_{i}\right)-y_{i}\right)^{2}
\]


```r
EF <- sum((nn.out.1$fitted.values-ozone$O3)^2)
EF  
```

```
## [1] 21115.41
```

Compute the generated RSS representing the unexplained variation if we consider no input features 

\[
\text{RSS}=\sum_{i=1}^{n}\left(y_{i}-\bar{y}\right)^{2}
\]


```r
RSS  <- sum((ozone$O3-mean(ozone$O3))^2)
RSS
```

```
## [1] 21115.41
```

How EF and RSS compare?  

You may observe that they are close --thats bad news; WHY?.  


```r
EF  
```

```
## [1] 21115.41
```

```r
RSS
```

```
## [1] 21115.41
```

```r
# Well they are close, so our naively trained NN has no predictive ability. 
#This is possibly due to a careless training of the NN, aka, the estimated weights are not good choices. 
#This is because training/learning NN is a non-convex learning problem, aka there are many local minima, some of them away from the global minimum. 
#I need to find a way to discover better values for the weights
# It can be done by standardizing the dataset values, and training the NN multiple times with different starting values for the weights (aka seeds)  
```


## Task: Standardise inputs / outputs (given)

The problem with the above NN, may be that, in the nnet R package, the seeds (starting values) in the SGD used to learn the weights of the NN were "bad".  

We can try to run the learning procedure multiple times each time with different SGD seeds for the weights. This may be facilitated if we standardize each variable in the data set.  

Observe that the examples of $x$ are in very different scales.  Run:  


```r
apply(ozone,2,sd)
```

```
##          O3          vh        wind    humidity        temp         ibh 
##    8.011277  105.708241    2.116963   19.865000   14.458737 1803.885870 
##         dpg         ibt         vis         doy 
##   35.717181   76.679424   79.362393  104.376374
```

Standardize the ozone{faraway} data to have mean zero and variance one, by using the command `scale{base}`.  

Save the rescaled data in the object "ozone.rescaled".   

Check again if they have really been standardized  

Do it below:  


```r
ozone.rescaled <- scale(ozone) 
apply(ozone.rescaled,2,mean)
```

```
##            O3            vh          wind      humidity          temp 
##  5.093779e-17  4.172967e-15 -6.994320e-17 -1.160701e-16 -2.048978e-16 
##           ibh           dpg           ibt           vis           doy 
## -5.620044e-17 -7.963268e-17  5.429619e-17  1.958784e-17 -1.169013e-16
```

```r
apply(ozone.rescaled,2,sd)
```

```
##       O3       vh     wind humidity     temp      ibh      dpg      ibt 
##        1        1        1        1        1        1        1        1 
##      vis      doy 
##        1        1
```

## Task: Standardise inputs / outputs (to be done in the computer practical class)

Now use the rescaled ozone data in the object "ozone.rescaled".  

Fit the NN $100$ times, each time using a different seed for the learning procedure.   

+ Essentially, code a for loop fitting again-and-again the NN  

Each time start with different seed for the SGD learning algorithm  

+ you can just use the command **set.seed( r )** before the command **nnet{nnet}** each time you fit the NN, where **r** is a different number each time.  

Among all the fitted NN, find the one that produces the smallest EF  

\[
\text{EF}\left(w|z\right)=\sum_{i=1}^{n}\left(h_{w}\left(x_{i}\right)-y_{i}\right)^{2}
\]

Save the produced object (output) of the corresponding nnet{nnet} call as your best fit in the object named as "nn.out.1.best".  

Do it below 


```r
Nrealizations <- 100
#
nn.out.1.best <- nnet(O3 ~ temp + ibh + ibt, ozone.rescaled, size=2, linout=T)
#EF.best <- sum((nn.out.1.best$fitted.values-ozone[$O3],1])^2)
EF.best <- nn.out.1.best$value
#
for (r in 1:Nrealizations) {
  #
  set.seed( r )
  #
  nn.out.1.new <- nnet(O3 ~ temp + ibh + ibt, ozone.rescaled, size=2, linout=T)
  EF.new <- nn.out.1.new$value
  #
  if (EF.new < EF.best) {
   #
    nn.out.1.best <- nn.out.1.new
    #
    EF.best <- EF.new
  }
}
```

Report your discovered best Error Function value EF, and compare it to the RSS computed earlier.   

\[
\text{RSS}=\sum_{i=1}^{n}\left(y_{i}-\bar{y}\right)^{2}
\]

Is it better now?   

Do it below


```r
EF.best
```

```
## [1] 88.53709
```

```r
RSS  <- sum((ozone$O3-mean(ozone$O3))^2)
RSS
```

```
## [1] 21115.41
```
## Task: Print the estimated weights (to be done in the computer practical class)

Print the estimated weights of the fitted feed forward neural network.  

Use the command **summary{base}** to do this  as "summary( output object from nnet function )" 

Do it below


```r
summary(nn.out.1.best)
```

```
## a 3-2-1 network with 11 weights
## options were - linear output units 
##  b->h1 i1->h1 i2->h1 i3->h1 
## -56.93  29.51 -99.99 -55.70 
##  b->h2 i1->h2 i2->h2 i3->h2 
##   1.14  -0.96   0.83   0.28 
##   b->o  h1->o  h2->o 
##   3.36  -0.69  -4.51
```
Description of what you have gotten:  

+ i2->h1, refers to the link between the second input variable and the first hidden neuron.  

+ b refers to the bias, 

+ o refers to the output,  etc...  

+ b->o refers to the weight on a NN edge linking an input neuron (constant input equalt to one) and an aoutput neuron and skipping the hidden layer. This is called skipping weight. This can be done, it is allowed in the FFNN, although it is often avoided. 


## Task: Plot predictions (given)

Plot the predicted "O3" values for "temp" in the range $(-3,3)$, while the other two input variables **ibh** and **ibt** are fixed to points **ibh=0**, **ibt=0**.  

+ Remember that your NN is trained on the standardized training data set.  

  + You can get the mean and variance by using  
  
    + attributes(ozone.rescaled)$"scaled:center" and 
  
    + attributes(ozone.rescaled)$"scaled:scale"  

+ Create a data frame from all combinations of the supplied vectors; that is **temp** in the range $(-3,3)$, **ibh=0**, **ibt=0**.  

  + Here, use the command **expand.grid {base}" as "expand.grid(temp=seq(-3,3,0.1),ibh=0,ibt=0)**  
  
+ Create the x-axis values of the plot in the range $(-3,3)$ and re-scale them back to the original scale
    
+ Create the y-axis values of prediction of the plot and re-scale them back to the original scale  

  + you can use the command **predict**; check **?predict.nnet**  

See the given code below to as you may use it for the next two tasks:   



```r
#
# Create the input object x that will be used in the function predict() 
#
xx <- expand.grid(temp=seq(-3,3,0.1),ibh=0,ibt=0)
#
# Make the predictions
#
pred.1.best <- predict(nn.out.1.best,new=xx)
#
# Your trained model is scaled, you need to bring its unites (input/output) back to the natural scale
#
# Here is how you get the mean and varances of the original data set
#
ozmeans <- attributes(ozone.rescaled)$"scaled:center"
ozscales <- attributes(ozone.rescaled)$"scaled:scale"
#
# Apply the re-scaling back to the original data for the inputs
#
xx.rescaled <- xx$temp*ozscales['temp']+ozmeans['temp']
#
# Apply the re-scaling back to the original data for the inputs
#
pred.1.best.rescaled <- pred.1.best*ozscales['O3']+ozmeans['O3']
#
# plot
#
plot(xx.rescaled,
     pred.1.best.rescaled,
     cex=2,xlab="Temp",ylab="O3")
```

![plot of chunk unnamed-chunk-19](figure/unnamed-chunk-19-1.png)

## Task: Plot predictions (to be done in the computer practical class)

Plot the predicted "O3" values for **ibh** in the range $(-3,3)$, while the other two input variables **ibt** and **temp** are fixed to points **ibh=0**, **ibt=0**.  

Essentially, do the same as above but with for **ibh** instead of **temp**, i.e. use 

+ **xx <-expand.grid(temp=0,ibh=seq(-3,3,0.1),ibt=0)**  


```r
#
# Create the input object x that will be used in the function predict() 
#
xx <-expand.grid(temp=0,ibh=seq(-3,3,0.1),ibt=0)
#
# Make the predictions
#
pred.1.best <- predict(nn.out.1.best,new=xx)
#
# Your trained model is scaled, you need to bring its unites (input/output) back to the natural scale
#
# Here is how you get the mean and varances of the original data set
#
ozmeans <- attributes(ozone.rescaled)$"scaled:center"
ozscales <- attributes(ozone.rescaled)$"scaled:scale"
#
# Apply the re-scaling back to the original data for the inputs
#
xx.rescaled <- xx$ibh*ozscales['ibh']+ozmeans['ibh']
#
# Apply the re-scaling back to the original data for the inputs
#
pred.1.best.rescaled <- pred.1.best*ozscales['O3']+ozmeans['O3']
#
# plot
#
plot(xx.rescaled,
     pred.1.best.rescaled,
     cex=2,xlab="ibh",ylab="O3")
```

![plot of chunk unnamed-chunk-20](figure/unnamed-chunk-20-1.png)

## Task: Plot predictions (to be done in the computer practical class)

Plot the predicted "O3" values for **ibt** in the range $(-3,3)$, while the other two input variables **ibh** and **temp** are fixed to points **ibh=0**, **ibt=0**.  

Essentially, copy / past the code above and swap **ibt** and **temp**, i.e. use 

+ xx <-expand.grid(temp=0,ibh=0,ibt=seq(-3,3,0.1))  


```r
#
# Create the input object x that will be used in the function predict() 
#
xx <-expand.grid(temp=0,ibh=0,ibt=seq(-3,3,0.1))  
#
# Make the predictions
#
pred.1.best <- predict(nn.out.1.best,new=xx)
#
# Your trained model is scaled, you need to bring its unites (input/output) back to the natural scale
#
# Here is how you get the mean and varances of the original data set
#
ozmeans <- attributes(ozone.rescaled)$"scaled:center"
ozscales <- attributes(ozone.rescaled)$"scaled:scale"
#
# Apply the re-scaling back to the original data for the inputs
#
xx.rescaled <- xx$ibt*ozscales['ibt']+ozmeans['ibt']
#
# Apply the re-scaling back to the original data for the inputs
#
pred.1.best.rescaled <- pred.1.best*ozscales['O3']+ozmeans['O3']
#
# plot
#
plot(xx.rescaled,
     pred.1.best.rescaled,
     cex=2,xlab="ibt",ylab="O3")
```

![plot of chunk unnamed-chunk-21](figure/unnamed-chunk-21-1.png)

## Task: Perform the training with shrinckage  (to be done in the computer practical class)  

The observed  discontinuities in the plots may possibly be due to the unreasonably large weights in the NN.  

Plain NN training tend to produce large weights in order to optimize the fit against the training data set, but the predictions will be unstable, especially for extrapolation.  

To address the above one can implement shrinkage methods, eg Ridge  

\[
w^{*} =\underset{w\in\mathcal{H}}{\arg\min}\left(R_{g}\left(w\right)+\lambda|w|_2^2\right)\label{eq:dsghadfhafdb-1}
\]
\[
w^{*} =\underset{w\in\mathcal{H}}{\arg\min}\left(\text{E}_{z\sim g}\left(\ell\left(w,z\right)+\lambda|w|_2^2\right)\right)
\]

Fit the NN $100$ times, each time using different seeds for the SGD (as above).  

You can use a Ridge shrinkage penalty, by using the argument **decay** in the function **nnet{nnet}**.  

+ Try decay=$0.001$.

Save the produced output object of the corresponding **nnet{nnet}** call as your best fit in the object "nn.out.1.decay.best".  

Do it below 



```r
Nrealizations <- 100
#
nn.out.1.decay.best <- nnet(O3 ~ temp + ibh + ibt, ozone.rescaled, size=2, linout=T, decay=0.001)
#EF.best <- sum((nn.out.1.best$fitted.values-ozone[$O3],1])^2)
EF.decay.best <- nn.out.1.best$value
#
for (r in 1:Nrealizations) {
  #
  set.seed( r )
  #
  nn.out.1.new <- nnet(O3 ~ temp + ibh + ibt, ozone.rescaled, size=2, linout=T, decay=0.001)
  EF.new <- nn.out.1.new$value
  #
  if (EF.new < EF.best) {
   #
    nn.out.1.decay.best <- nn.out.1.new
    #
    EF.decay.best <- EF.new
  }
}
```


## Task: Print the weights 

Print the produced Error Function, and compare them to those you computed without the decay. Why do you think you observe this?    

Print the estimated weights, and compare them to those you computed without the decay.  

Do it below.  


```r
EF.decay.best
```

```
## [1] 88.53709
```


```r
summary(nn.out.1.decay.best)
```

```
## a 3-2-1 network with 11 weights
## options were - linear output units  decay=0.001
##  b->h1 i1->h1 i2->h1 i3->h1 
##   0.94   0.84   1.22  -0.55 
##  b->h2 i1->h2 i2->h2 i3->h2 
##   0.47   0.65   1.38  -0.82 
##  b->o h1->o h2->o 
## -1.32  8.90 -8.21
```

## Task: Prediction plots  (given as a homework)

Plot the predicted, **O3** values for **ibt** in the range $(-3,3)$, while the other two input variables are fixed to points **ibh=0**, **temp=0**.

Essentially, do the same as above but with for **ibt** instead of **terms**, i.e. use 

+ **xx <-expand.grid(temp=0,ibh=0,ibt=seq(-3,3,0.1))**  

Essentially copy / paste your code above and implemented on "nn.out.1.decay.best" instead on "nn.out.1.best"

Do it below.


```r
#
# Create the input object x that will be used in the function predict() 
#
xx <-expand.grid(temp=0,ibh=0,ibt=seq(-3,3,0.1))  
#
# Make the predictions
#
pred.1.best <- predict(nn.out.1.decay.best,new=xx)
#
# Your trained model is scaled, you need to bring its unites (input/output) back to the natural scale
#
# Here is how you get the mean and varances of the original data set
#
ozmeans <- attributes(ozone.rescaled)$"scaled:center"
ozscales <- attributes(ozone.rescaled)$"scaled:scale"
#
# Apply the re-scaling back to the original data for the inputs
#
xx.rescaled <- xx$ibt*ozscales['ibt']+ozmeans['ibt']
#
# Apply the re-scaling back to the original data for the inputs
#
pred.1.best.rescaled <- pred.1.best*ozscales['O3']+ozmeans['O3']
#
# plot
#
plot(xx.rescaled,
     pred.1.best.rescaled,
     cex=2,xlab="ibt",ylab="O3",
     type="l")
```

![plot of chunk unnamed-chunk-25](figure/unnamed-chunk-25-1.png)


You should observe that the line is now smoother as it was supposed to be.  


---  

---  


## Additional tasks  

### Multi-class classification problem    

How would you modify the above code to address a multi-class classification problem?  

Practice on the following variation.  

+ [LINK TO TASKS](http://htmlpreview.github.io/?https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023/blob/main/Computer_practical/Artificial_Neural_Networks_Artificial_Neural_Networks_MultiClassClassification_tasks.nb.html)  

+ [LINK TO SOLUTIONS](http://htmlpreview.github.io/?https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023/blob/main/Computer_practical/Artificial_Neural_Networks_Artificial_Neural_Networks_MultiClassClassification_solutions.nb.html)  



