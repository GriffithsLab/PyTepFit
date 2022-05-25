# Modelling large-scale brain network dynamics underlying the TMS-EEG evoked response

This repository includes the code required to reproduce the results in: "TMS-EEG evoked responses are driven by recurrent large-scale network dynamics" Davide Momi, Zheng Wang, John D. Griffiths



## Study Overview:
Grand-mean structural connectome was calculated using neuroimaging data of 400 healthy young individuals (Males= 170; Age range 21-35 years old) from the Human Connectome Project (HCP) Dataset (https://ida.loni.usc.edu) The Schaefer Atlas of 200 parcels was used, which groups brain regions into 7 canonical functional networks (Visual: VISN, Somato-motor: SMN, Dorsal Attention: DAN, Anterior Salience: ASN, Limbic: LIMN, Fronto-parietal: FPN, Default Mode DMN). These parcels were mapped to the individual’s FreeSurfer parcellation using spherical registration. Once this brain parcellation covering cortical structures was extrapolated, it was then used to extract individual structural connectomes. The Jansen-Rit model 22, a physiological connectome-based neural mass model, was embedded in every parcel for simulating time series. The TMS-induced polarization effect of the resting membrane potential was modeled by perturbing voltage offset uTMS to the mean membrane potential of the excitatory interneurons. A lead-field matrix was then used for moving the parcels’ timeseries into channel space and generating simulated TMS-EEG activity. The goodness-of-fit (loss) was calculated as the spatial distance between simulated and empirical TMS-EEG timeseries from an open dataset. The corresponding gradient was computed using Automatic Differentiation. Model parameters were then updated accordingly using the optimization algorithm (ADAM). The model was run with new updated parameters, and the process was repeated until convergence between simulated and empirical timeseries was found. At the end of iteration, the optimal model parameters were used to generate the fitted simulated TMS-EEG activity, which was compared with the empirical data at both the channel and source level

![alt text](https://github.com/GriffithsLab/PyTepFit/blob/master/img/Figure_1.png)


## Conceptual Framework:
Studying the role of recurrent connectivity in stimulation-evoked neural responses with computational models. Shown here is a schematic overview of the hypotheses, methodology, and general conceptual framework of the present work. A) Single TMS stimulation (i [diagram], iv [real data]) pulse applied to a target region (in this case left M1) generates an early response (TEP waveform component) at EEG channels sensitive to that region and its immediate neighbours (ii). This also appears in more distal connected regions such as the frontal lobe (iii) after a short delay due to axonal conduction and polysynaptic transmission. Subsequently, second and sometimes third late TEP components are frequently observed at the target site (i, iv), but not in nonconnected regions (v). Our question is whether these late responses represent evoked oscillatory ‘echoes’ of the initial stimulation that are entirely ‘local’ and independent of the rest of the network, or whether they rather reflect a chain of recurrent activations dispersing from and then propagating back to the initial target site via the connectome. B) In order to investigate this, precisely timed communication interruptions or ‘virtual lesions’ (vii) are introduced into an otherwise accurate and well-fitting computational model of TMS-EEG stimulation responses (vi). and the resulting changes in the propagation pattern (viii) were evaluated. No change in the TEP component of interest would support the ‘local echo’ scenario, and whereas suppressed TEPs would support the ‘recurrent activation’ scenario.


![alt text](https://github.com/GriffithsLab/PyTepFit/blob/master/img/Figure_2.PNG)


**Dependencies**   

* [`pytorch`](https://pytorch.org/)
* [`mne`](https://mne.tools/stable/index.html)
* [`numpy`](https://numpy.org/)
* [`scipy`](https://scipy.org/scipylib/index.html)
* [`scikit-learn`](https://scikit-learn.org/stable/)
* [`matplotlib`](https://matplotlib.org/)
* [`nibabel`](https://nipy.org/nibabel/index.html)
* [`nilearn`](https://nilearn.github.io/)

## Data   

For the purposes of this study, already preprocessed TMS-EEG data following a stimulation of primary motor cortex (M1) of twenty healthy young individuals (24.50 ± 4.86 years; 14 females) were taken from an open dataset (https://figshare.com/articles/dataset/TEPs-_SEPs/7440713). For details regarding on the data acquisition and the preprocessing steps please refer to the original paper

* Biabani M, Fornito A, Coxon JP, Fulcher BD & Rogasch NC (2021). The correspondence between EMG and EEG measures of changes in cortical excitability following transcranial magnetic stimulation. J Physiol 599, 2907–2932.
* Biabani M, Fornito A, Mutanen TP, Morrow J & Rogasch NC (2019). Characterizing and minimizing the contribution of sensory inputs to TMS-evoked potentials. Brain Stimulat 12, 1537–1552.



## Notebooks

|   | Run | View |
| - | --- | ---- |
| Fitting individual subjects’ empirical TEPs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Davi1990/PyTepFit/blob/master/nb/model_fit.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/Davi1990/PyTepFit/blob/master/nb/model_fit.ipynb?flush_cache=true) |


|   | Run | View |
| - | --- | ---- |
| Goodness of Fit | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Davi1990/PyTepFit/blob/master/nb/Goodness_of_fit.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/Davi1990/PyTepFit/blob/master/nb/model_fit.ipynb?flush_cache=true) |


Comparison between simulated and empirical TMS-EEG data in channel space. A) Empirical and simulated TMSEEG time series for 3 representative subjects showing a robust recovery of individual subjects’ empirical TEP propagation patterns in model-generated activity EEG time series. B) Pearson correlation coefficients over subjects between empirical and simulated TMS-EEG time series. C) Time-wise permutation tests results showing the significant Pearson correlation coefficient and the corresponding reversed p-values (bottom) for every electrode. D) PCI values extracted from the empirical (orange) and simulated (blue) TMS-EEG time series (left). Significant positive correlation (R2 = 80%, p < 0.0001) was found between the simulated and the empirical PCI (right).

![alt text](https://github.com/GriffithsLab/PyTepFit/blob/master/img/Figure_3.PNG)


Comparison between simulated and empirical TMS-EEG data in source space. A) TMS-EEG time series showing a robust recovery of grand-mean empirical TEP patterns in model-generated EEG time series. B) Source reconstructed TMS-evoked propagation pattern dynamics for empirical (top) and simulated (bottom) data. C) Time-wise permutation test results showing the significant Pearson correlation coefficient (tp) and the corresponding reversed p-values (bottom) for every vertex. Network-based dSPM values extracted for the grand mean empirical (left) and simulated (right) source-reconstructed time series.

![alt text](https://github.com/GriffithsLab/PyTepFit/blob/master/img/Figure_4.PNG)


|   | Run | View |
| - | --- | ---- |
| Params exploration | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Davi1990/PyTepFit/blob/master/nb/Params_exploration.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/Davi1990/PyTepFit/blob/master/nb/model_fit.ipynb?flush_cache=true) |


Synaptic time constant of inhibitory population affects early and late TEP amplitudes. A) Singular value decomposition (SVD) topoplots for simulated (top) and empirical (bottom) TMS-EEG data. By looking at simulated (top) and empirical (bottom) grand mean timeseries, results revealed that the first (orange outline) and the second (blue outline) components were located ∼65ms and ∼110ms after the TMS pulse. B) First (orange outline) and second (blue outline) components latency and amplitude were extracted for every subject and the distribution plots (top row) show the time location where higher cosine similarity with the SVD components were found. Scatter plots (bottom row) show a negative (left) and positive (right) significant correlation between the synaptic time constant of the inhibitory population and the amplitude of the the first and second component. C) Model-generated first and second SVD components for 2 representative subjects with high (top) and low (bottom) value for the synaptic time constant of the inhibitory population. Topoplots show that higher synaptic time constant affects the amplitude of the individual first and second SVD components. D) Model-generated TMS-EEG data were run using the optimal (top right) or a value decreased by 85% (central left) for the synaptic time constant of the inhibitory population. Absolute values for different values of the synaptic time constant of the inhibitory population (bottom right). Results show an increase in the amplitude of the early first local component and a decrease of the second latest global component

![alt text](https://github.com/GriffithsLab/PyTepFit/blob/master/img/Figure_6.PNG)
