# Multipole Expansion

This is the code repository for reproducing the results of the paper:
*Multipole expansion for HI intensity mapping experiments: simulations and modelling*
> [https://arxiv.org/abs/2002.05626]

## Code developers:

**Steve Cunnington**, **Paula Soares**, **Alkistis Pourtsidou**, and **Chris Blake**.

## MultiDark Data:

> http://www.multidark.org/

This project made use of data from the MultiDark simulations. This has all been condensed into premade HI intensity maps with corresponding foreground contaminants and is all available in the **MultiDarkSims** folder. Please refer to our paper for more details on the creation of this data and relevant references.

>These maps were run with the Queen Mary’s
Apocrita HPC facility, supported by QMUL Research-IT
http://doi.org/10.5281/zenodo.438045 using Python 3.6.5.

## Running the Code:

For the main MultiDark simulation results we provide three run scripts and provide their associated Figure numbers in relation to the companion paper:
 - runMainMultipoles.py [Figures 7, 11 & 12]
    - Set `DoWedges = True` (for Figure 7)
    - Set `DoWedges = False`
      - & `zeff = 2.03` (for Figure 11)
      - & `zeff = 0.82` (for Figure 12)    
 - runNoRSDMultipoles.py [Figure 10]
 - runkperpkpara.py [Figures 8 & 9]

which can be simply executed with `python [script-name].py`

Using the available map data contained in the **MultiDarkSims** folder and available function in rest of the repo, this should produce the main results from the paper.

Where relevant, error bars are produced using the default theoretical setting. However the Jackknifing technique can be turned on by setting `Jackknife = True`

### Lognormal and MCMC Codes:

For the MCMC analysis in the paper, we used lognormal mocks generated using `nbodykit`. Scripts for how to generate these lognormal mocks and for how to run the MCMC analysis can be found in the **JupyterNotebooks** folder.

 - `GeneratingLognormalMocks.ipynb`:
    - runs through how to generate a lognormal mock catalog of objects using `nbodykit`, smooth the overdensity field and add and remove foregrounds.

 - `MonopoleMCMCAnalysis.ipynb` [Figure 14]

## Packages Required:

 - Python3
 - numpy
 - matplotlib
 - scipy
 - astropy
 - sklearn
 - jupyter [only needed for running notebooks from **JupyterNotebooks** folder]
 - nbodykit [only needed for `GeneratingLognormalMocks.ipynb`]
 - emcee [only needed for `MonopoleMCMCAnalysis.ipynb`]

## Credit:

If you use these codes in your research, we kindly ask
that you cite this repository [https://github.com/IntensityTools/MultipoleExpansion] and the following papers:

```
@article{Cunnington:2020mnn,
    author = "Cunnington, Steven and Pourtsidou, Alkistis and Soares, Paula S. and Blake, Chris and Bacon, David",
    title = "{Multipole expansion for H i intensity mapping experiments: simulations and modelling}",
    eprint = "2002.05626",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1093/mnras/staa1524",
    journal = "Mon. Not. Roy. Astron. Soc.",
    volume = "496",
    number = "1",
    pages = "415--433",
    year = "2020"
}
```

```
@article{Blake:2019ddd,
    author = "Blake, Chris",
    title = "{Power spectrum modelling of galaxy and radio intensity maps including observational effects}",
    eprint = "1902.07439",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1093/mnras/stz2145",
    journal = "Mon. Not. Roy. Astron. Soc.",
    volume = "489",
    number = "1",
    pages = "153--167",
    year = "2019"
}
```

## Contact:

For any questions or comments please contact:<br/>
**Steve Cunnington** [s.cunnington@qmul.ac.uk]<br/>
**Alkistis Pourtsidou** [a.pourtsidou@qmul.ac.uk]

Feel free to contact us for any comments, suggestions, or bug reports or please open a **New issue** from the Issues tab.
