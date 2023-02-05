<!-- -------------------------------------------------------------------------------- -->

<!-- Copyright 2021 Georgios Karagiannis -->

<!-- georgios.karagiannis@durham.ac.uk -->
<!-- Associate Professor -->
<!-- Department of Mathematical Sciences, Durham University, Durham,  UK  -->

<!-- This file is part of Bayesian_Statistics_Michaelmas_2021 (MATH3341/4031 Bayesian Statistics III/IV) -->
<!-- which is the material of the course (MATH3341/4031 Bayesian Statistics III/IV) -->
<!-- taught by Georgios P. Katagiannis in the Department of Mathematical Sciences   -->
<!-- in the University of Durham  in Michaelmas term in 2019 -->

<!-- Bayesian_Statistics_Michaelmas_2021 is free software: you can redistribute it and/or modify -->
<!-- it under the terms of the GNU General Public License as published by -->
<!-- the Free Software Foundation version 3 of the License. -->

<!-- Bayesian_Statistics_Michaelmas_2021 is distributed in the hope that it will be useful, -->
<!-- but WITHOUT ANY WARRANTY; without even the implied warranty of -->
<!-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the -->
<!-- GNU General Public License for more details. -->

<!-- You should have received a copy of the GNU General Public License -->
<!-- along with Bayesian_Statistics_Michaelmas_2021  If not, see <http://www.gnu.org/licenses/>. -->

<!-- -------------------------------------------------------------------------------- -->

<!-- -------------------------------------------------------------------------------- -->

<!-- Copyright 2019 Georgios Karagiannis -->

<!-- georgios.karagiannis@durham.ac.uk -->
<!-- Assistant Professor -->
<!-- Department of Mathematical Sciences, Durham University, Durham,  UK  -->

<!-- This file is part of Bayesian_Statistics (MATH3341/4031 Bayesian Statistics III/IV) -->
<!-- which is the material of the course (MATH3341/4031 Bayesian Statistics III/IV) -->
<!-- taught by Georgios P. Katagiannis in the Department of Mathematical Sciences   -->
<!-- in the University of Durham  in Michaelmas term in 2019 -->

<!-- Bayesian_Statistics is free software: you can redistribute it and/or modify -->
<!-- it under the terms of the GNU General Public License as published by -->
<!-- the Free Software Foundation version 3 of the License. -->

<!-- Bayesian_Statistics is distributed in the hope that it will be useful, -->
<!-- but WITHOUT ANY WARRANTY; without even the implied warranty of -->
<!-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the -->
<!-- GNU General Public License for more details. -->

<!-- You should have received a copy of the GNU General Public License -->
<!-- along with Bayesian_Statistics  If not, see <http://www.gnu.org/licenses/>. -->

<!-- -------------------------------------------------------------------------------- -->


Aim
===

The aim of the computer practical handouts is for the students to be able to implement the methods introduced in the lecture int he computing environment and particularly in R programming language. 

------------------------------------------------------------------------

Preview:
========

Stochastic learning methods (Computer practical 1)

-   Tasks : [Rnotebook](http://htmlpreview.github.io/?https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023/blob/main/Computer_practical/Stochastic_learning_methods_tasks.nb.html) & [Rmd](https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023/blob/main/Computer_practical/Stochastic_learning_methods_tasks.Rmd)  

-   Complete : [Rnotebook](http://htmlpreview.github.io/?https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023/blob/main/Computer_practical/Stochastic_learning_methods.nb.html) & [Rmd](https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023/blob/main/Computer_practical/Stochastic_learning_methods.Rmd)  

Stochastic learning methods SGLD (Computer practical 2)

-   Tasks : [Rnotebook](http://htmlpreview.github.io/?https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023/blob/main/Computer_practical/Stochastic_gradient_langevin_dynamics_tasks.nb.html) & [Rmd](https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023/blob/main/Computer_practical/Stochastic_gradient_langevin_dynamics_tasks.Rmd)  

-   Complete : [Rnotebook](http://htmlpreview.github.io/?https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023/blob/main/Computer_practical/Stochastic_gradient_langevin_dynamics.nb.html) & [Rmd](https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023/blob/main/Computer_practical/Stochastic_gradient_langevin_dynamics.Rmd)  

------------------------------------------------------------------------

Reference list
==============

*The material below is not examinable material, but it contains
references that students can follow if they want to further explore the
concepts introduced.*

-   References for *RJAGS*:
    -   [JAGS homepage](http://mcmc-jags.sourceforge.net)  
    -   [JAGS R CRAN
        Repository](https://cran.r-project.org/web/packages/rjags/index.html)  
    -   [JAGS Reference
        Manual](https://cran.r-project.org/web/packages/rjags/rjags.pdf)  
    -   [JAGS user
        manual](https://sourceforge.net/projects/mcmc-jags/files/Manuals/4.x/jags_user_manual.pdf/download)
-   Reference for *R*:
    -   [Cheat sheet with basic commands for
        *R*](https://www.rstudio.com/wp-content/uploads/2016/10/r-cheat-sheet-3.pdf)
-   Reference of *rmarkdown* (optional)
    -   [R Markdown
        cheatsheet](https://www.rstudio.com/wp-content/uploads/2016/03/rmarkdown-cheatsheet-2.0.pdf)  
    -   [R Markdown Reference
        Guide](http://442r58kc8ke1y38f62ssb208-wpengine.netdna-ssl.com/wp-content/uploads/2015/03/rmarkdown-reference.pdf)  
    -   [knitr options](https://yihui.name/knitr/options)
-   Reference for *Latex* (optional):
    -   [Latex Cheat
        Sheet](https://wch.github.io/latexsheet/latexsheet-a4.pdf)

------------------------------------------------------------------------

Setting up the computing environment
====================================

### CIS computers

From AppHub, load the modules:

1.  Google Chrome

2.  LaTex

3.  rstudio

### Your personal computers (Do not do it on CIS computers)

There is not need to do this in CIS computers as the required foftware
is (supposed to be) properly installed.

The instructions below are at your own risk…

We recommend the use of LINUX operation system.

Briefly, you need to do the following:

1.  Install LaTex (optional but recommended)
    -   Source: download it from
        <https://www.tug.org/texlive/acquire-netinstall.html>
    -   Debian: *apt-get install texlive-full*  
    -   Fedora: *yum install texlive texlive-latex*  
    -   windows: download it from
        <https://miktex.org/howto/install-miktex>
    -   macos: download it from <https://www.tug.org/mactex/>
2.  Install R computing environment version R 2.14.0 or later.
    -   Source: download it from here: <https://cran.r-project.org/>  
    -   Debian: *sudo apt install r-base*  
    -   Fedora: *yum install -y R*  
    -   windows: download it from <https://cran.r-project.org/>
        -   I recomend tyou to install *Rtools* as well, for you to be
            able to instal R packages.  
    -   macos: download it from <https://www.tug.org/mactex/>
3.  Install the latest Rstudio (recommended)
    -   Any OS: Download it from here:
        <https://www.rstudio.com/products/rstudio/download/>

------------------------------------------------------------------------


### How to download and use it in Rstudio cloud 

1. Go to the website [ <https://rstudio.cloud> ] , if you already have an account log in, otherwise register and then log in.  

2. After logging in,  
    
    1. go to Projects tab, 
    
    2. click on the *v* next to the *New Project* button to expand the pop-up menu list  
    
    3. click on the choice *New Project from Git Repo*  
    
    4. in the *URL of your Git repository* section insert the link: 
        
        <https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023.git> 

    ... this will gonna download the whole Machine learning and Neural Networks learning teaching material. You can navigate to the material.  

### How to download the whole repository

Ways:

1. You can click [[HERE](https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023/archive/refs/heads/main.zip)].

2. You can click the green button 'Clone or download' and download it as a zip file

3. You can use the program 'git' (<https://git-scm.com/>):
    
    -   in windows/linux: 
    
        download and install git from https://git-scm.com/
    
    -   in Debian linux run in the terminal: 
    
        sudo apt-get install git
    
    -   in Red Hat linux run in the terminal: 
    
        sudo yum install git
    
    ... then run:

    -   git clone https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023.git

4. You can use rstudio:

    1.  Go to File &gt; New Project &gt; Version Control &gt; Git
    
    2.  In the section *Repository URL* write
        
        -   <https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023.git>
        
        -   … and complete the rest as you wish
    
    3.  Hit *Create a Project*

### How to download a specific folder only

1. In install [Firefox GitZip add-on](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=2ahUKEwias52xjd3nAhXPUs0KHeXHCEUQFjACegQIAhAB&url=https%3A%2F%2Faddons.mozilla.org%2Fen-US%2Ffirefox%2Faddon%2Fgitzip%2F&usg=AOvVaw37servrJ29tuNcx9dIQDqy) or the [Chrome GitZip add-on](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwias52xjd3nAhXPUs0KHeXHCEUQFjABegQIARAB&url=https%3A%2F%2Fchrome.google.com%2Fwebstore%2Fdetail%2Fgitzip-for-github%2Fffabmkklhbepgcgfonabamgnfafbdlkn%3Fhl%3Den&usg=AOvVaw1Pn3VXuXz1Fphl7dsPEhDS)  

2. Double click on the items you need.  

3. Click download button at bottom-right.  

4. See the progress dashboard and wait for browser trigger download.  

5. Get the ZIP file.  

<!--

Ways:

1. You can use the GitZip add-on for Firefox available [HERE](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=2ahUKEwias52xjd3nAhXPUs0KHeXHCEUQFjACegQIAhAB&url=https%3A%2F%2Faddons.mozilla.org%2Fen-US%2Ffirefox%2Faddon%2Fgitzip%2F&usg=AOvVaw37servrJ29tuNcx9dIQDqy) or the Chrome add-on GitZip available [HERE](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwias52xjd3nAhXPUs0KHeXHCEUQFjABegQIARAB&url=https%3A%2F%2Fchrome.google.com%2Fwebstore%2Fdetail%2Fgitzip-for-github%2Fffabmkklhbepgcgfonabamgnfafbdlkn%3Fhl%3Den&usg=AOvVaw1Pn3VXuXz1Fphl7dsPEhDS)

    1. In install [Firefox GitZip add-on](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=2ahUKEwias52xjd3nAhXPUs0KHeXHCEUQFjACegQIAhAB&url=https%3A%2F%2Faddons.mozilla.org%2Fen-US%2Ffirefox%2Faddon%2Fgitzip%2F&usg=AOvVaw37servrJ29tuNcx9dIQDqy) or the [Chrome GitZip add-on](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwias52xjd3nAhXPUs0KHeXHCEUQFjABegQIARAB&url=https%3A%2F%2Fchrome.google.com%2Fwebstore%2Fdetail%2Fgitzip-for-github%2Fffabmkklhbepgcgfonabamgnfafbdlkn%3Fhl%3Den&usg=AOvVaw1Pn3VXuXz1Fphl7dsPEhDS)  

    2. Double click on the items you need.  
    
    3. Click download button at bottom-right.  
    
    4. See the progress dashboard and wait for browser trigger download.  
    
    5. Get the ZIP file.  

2. You can use 'git' (<https://git-scm.com/>). 

    E.g., assume you wish to download the sub-folder 'Computer_practical':

    -   run in the terminal the following:
        
        *mkdir Machine_Learning_and_Neural_Networks_III_Epiphany_2023  
        cd Machine_Learning_and_Neural_Networks_III_Epiphany_2023  
        git init  
        git remote add -f origin https://github.com/georgios-stats/Machine_Learning_and_Neural_Networks_III_Epiphany_2023.git  
        git config core.sparseCheckout true  
        echo "Computer_practical/*" >> .git/info/sparse-checkout  
        git pull origin master*
        
-->

### How to download a specific file

1. You can just navigate to the file from the browser and download it.

2. You can use the GitZip add-on for Firefox available [HERE](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=2ahUKEwias52xjd3nAhXPUs0KHeXHCEUQFjACegQIAhAB&url=https%3A%2F%2Faddons.mozilla.org%2Fen-US%2Ffirefox%2Faddon%2Fgitzip%2F&usg=AOvVaw37servrJ29tuNcx9dIQDqy) or the Chrome add-on GitZip available [HERE](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwias52xjd3nAhXPUs0KHeXHCEUQFjABegQIARAB&url=https%3A%2F%2Fchrome.google.com%2Fwebstore%2Fdetail%2Fgitzip-for-github%2Fffabmkklhbepgcgfonabamgnfafbdlkn%3Fhl%3Den&usg=AOvVaw1Pn3VXuXz1Fphl7dsPEhDS)


------------------------------------------------------------------------
