# Modelling large-scale brain network dynamics underlying the TMS-EEG evoked response

This repository includes the code required to reproduce the results in: "TMS-EEG evoked responses are driven by recurrent large-scale network dynamics" Davide Momi, Zheng Wang, John D. Griffiths



## Study Overview:
Grand-mean structural connectome was calculated using neuroimaging data of 400 healthy young individuals (Males= 170; Age range 21-35 years old) from the Human Connectome Project (HCP) Dataset (https://ida.loni.usc.edu) The Schaefer Atlas of 200 parcels was used, which groups brain regions into 7 canonical functional networks (Visual: VISN, Somato-motor: SMN, Dorsal Attention: DAN, Anterior Salience: ASN, Limbic: LIMN, Fronto-parietal: FPN, Default Mode DMN). These parcels were mapped to the individual’s FreeSurfer parcellation using spherical registration. Once this brain parcellation covering cortical structures was extrapolated, it was then used to extract individual structural connectomes. The Jansen-Rit model 22, a physiological connectome-based neural mass model, was embedded in every parcel for simulating time series. The TMS-induced polarization effect of the resting membrane potential was modeled by perturbing voltage offset uTMS to the mean membrane potential of the excitatory interneurons. A lead-field matrix was then used for moving the parcels’ timeseries into channel space and generating simulated TMS-EEG activity. The goodness-of-fit (loss) was calculated as the spatial distance between simulated and empirical TMS-EEG timeseries from an open dataset. The corresponding gradient was computed using Automatic Differentiation. Model parameters were then updated accordingly using the optimization algorithm (ADAM). The model was run with new updated parameters, and the process was repeated until convergence between simulated and empirical timeseries was found. At the end of iteration, the optimal model parameters were used to generate the fitted simulated TMS-EEG activity, which was compared with the empirical data at both the channel and source level

![alt text](https://github.com/Davi1990/PyTepFit/blob/master/img/Figure_1.png)


## Conceptual Framework:
Studying the role of recurrent connectivity in stimulation-evoked neural responses with computational models. Shown here is a schematic overview of the hypotheses, methodology, and general conceptual framework of the present work. A) Single TMS stimulation (i [diagram], iv [real data]) pulse applied to a target region (in this case left M1) generates an early response (TEP waveform component) at EEG channels sensitive to that region and its immediate neighbours (ii). This also appears in more distal connected regions such as the frontal lobe (iii) after a short delay due to axonal conduction and polysynaptic transmission. Subsequently, second and sometimes third late TEP components are frequently observed at the target site (i, iv), but not in nonconnected regions (v). Our question is whether these late responses represent evoked oscillatory ‘echoes’ of the initial stimulation that are entirely ‘local’ and independent of the rest of the network, or whether they rather reflect a chain of recurrent activations dispersing from and then propagating back to the initial target site via the connectome. B) In order to investigate this, precisely timed communication interruptions or ‘virtual lesions’ (vii) are introduced into an otherwise accurate and well-fitting computational model of TMS-EEG stimulation responses (vi). and the resulting changes in the propagation pattern (viii) were evaluated. No change in the TEP component of interest would support the ‘local echo’ scenario, and whereas suppressed TEPs would support the ‘recurrent activation’ scenario.


![alt text](https://github.com/Davi1990/PyTepFit/tree/main/img/Figure_2.PNG)

## Large-scale neurophysiological brain network model:
A brain network model comprising 200 cortical areas was used to model TMS-evoked activity patterns, where each network node represents population-averaged activity of a single brain region according to the rationale of mean field theory60. We used the Jansen-Rit (JR) equations to describe activity at each node, which is one of the most widely used neurophysiological models for both stimulus-evoked and resting-state EEG activity measurements. JR is a relatively coarse-grained neural mass model of the cortical microcircuit, composed of three interconnected neural populations: pyramidal projection neurons, excitatory interneurons, and inhibitory interneurons. The excitatory and the inhibitory populations both receive input from and feed back to the pyramidal population but not to each other, and so the overall circuit motif contains one positive and one negative feedback loop. For each of the three neural populations, the post-synaptic somatic and dendritic membrane response to an incoming pulse of action potentials is described by the second-order differential equation:


<p align="center">
<img src=https://latex.codecogs.com/png.latex?%5CDdot%7Bv%7D%5Cleft%28t%5Cright%29%20&plus;%5Cfrac%7B2%7D%7B%5Ctau_%7Be%2Ci%7D%7D%5Cdot%7Bv%7D%5Cleft%28t%5Cright%29&plus;%5Cfrac%7B1%7D%7B%5Ctau_%7Be.i%7D%5E2%7Dv%5Cleft%28t%5Cright%29%20%3D%20%5Cfrac%7BH_%7Be%2Ci%7D%7D%7B%5Ctau_%7Be%2Ci%7D%7Dm%5Cleft%28t%5Cright%29>
</p>

which is equivalent to a convolution of incoming activity with a synaptic impulse response function 

<p align="center">
<img src=https://latex.codecogs.com/png.latex?v%28t%29%20%3D%20%5Cint%5Climits_0%5E%7B%5Cinfty%7D%20d%20%5Ctau%20m%28%5Ctau%29%20%5Ccdot%20h_%7Be%2Ci%7D%20%28t-%5Ctau%29>
</p>

whose kernel $h_{e,i}(t)$ is given by 

<p align="center">
<img src=https://latex.codecogs.com/png.latex?h_%7Be%2Ci%7D%20%3D%20%5Cfrac%7BH_%7Be%2Ci%7D%7D%7B%5Ctau_%7Be%2Ci%7D%7D%20%5Ccdot%20t%20%5Ccdot%20exp%28%20-%5Cfrac%7Bt%7D%7B%5Ctau_%7Be%2Ci%7D%7D%20%29>
</p>

where $m(t)$ is the (population-average) presynaptic input, $v(t)$ is the postsynaptic membrane potential, $H_{e,i}$ is the maximum postsynaptic potential and $\tau_{e,i}$ a lumped representation of delays occurring during the synaptic transmission. 

This synaptic response function, also known as a pulse-to-wave operator [Freeman et al., 1975](https://www.sciencedirect.com/book/9780122671500/mass-action-in-the-nervous-system), determines the excitability of the population, as parameterized by the rate constants $a$ and $b$, which are of particular interest in the present study. Complementing the pulse-to-wave operator for the synaptic response, each neural population also has wave-to-pulse operator [Freeman et al., 1975](https://www.sciencedirect.com/book/9780122671500/mass-action-in-the-nervous-system) that determines the its output - the (population-average) firing rate - which is an instantaneous function of the somatic membrane potential that takes the sigmoidal form 

<p align="center">
<img src=https://latex.codecogs.com/png.latex?Su%28t%29%20%3D%20%5Cbegin%7Bcases%7D%20%5Cfrac%7Be_%7B0%7D%7D%7B1-exp%28r%28v_%7B0%7D%20-%20v%28t%29%29%29%7D%20%26%20t%5Cgeq%200%20%5C%5C%200%20%26%20t%5Cleq%200%20%5C%5C%20%5Cend%7Bcases%7D>
</p>

where $e_{0}$ is the maximum pulse, $r$ is the steepness of the sigmoid function, and $v_0$ is the postsynaptic potential for which half of the maximum pulse rate is achieved.

In practice, we re-write the three sets of second-order differential equations that follow the form in (1) as pairs of coupled first-order differential equations, and so the full JR system for each individual cortical area $j \in i:N$ in our network of $N$=200 regions is given by the following six equations: 

<p align="center">
<img src=https://latex.codecogs.com/png.latex?%5Cdot%7Bv%7D_%7Bj1%7D%20%3D%20x_%7Bj1%7D>
</p>
<p align="center">
<img src=https://latex.codecogs.com/png.latex?%5Cdot%7Bx%7D_%7Bj1%7D%20%3D%20%5Cfrac%7BH_e%7D%7B%5Ctau_e%7D%5Cleft%28p%28t%29&plus;%5Cmathrm%7Bconn%7D_j&plus;S%28v_%7Bj2%7D%29%5Cright%29%20-%20%5Cfrac%7B2%7D%7B%5Ctau_e%7Dx_%7Bj1%7D%20-%20%5Cfrac%7B1%7D%7B%5Ctau_e%5E2%7Dv_%7Bj1%7D>
</p>
<p align="center">
<img src=https://latex.codecogs.com/png.latex?%5Cdot%7Bv%7D_%7Bj2%7D%20%3D%20x_%7Bj2%7D>
</p>
<p align="center">
<img src=https://latex.codecogs.com/png.latex?%5Cdot%7Bx%7D_%7Bj2%7D%20%3D%20%5Cfrac%7BH_i%7D%7B%5Ctau_i%7D%5Cleft%28S%28v_%7B3j%7D%29%5Cright%29%20-%20%5Cfrac%7B2%7D%7B%5Ctau_i%7Dx_%7Bj2%7D%20-%20%5Cfrac%7B1%7D%7B%5Ctau_i%5E2%7Dv_%7Bj2%7D>
</p>
<p align="center">
<img src=https://latex.codecogs.com/png.latex?%5Cdot%7Bv%7D_%7Bj3%7D%20%3D%20x_%7Bj3%7D>
</p>
<p align="center">
<img src=https://latex.codecogs.com/png.latex?%5Cdot%7Bx%7D_%7Bj3%7D%20%3D%20%5Cfrac%7BH_e%7D%7B%5Ctau_e%7D%5Cleft%28S%28v_%7Bj1%7D%20-%20v_%7Bj2%7D%29%5Cright%29%20-%20%5Cfrac%7B2%7D%7B%5Ctau_e%7Dx_%7Bj3%7D%20-%20%5Cfrac%7B1%7D%7B%5Ctau_e%5E2%7Dv_%7Bj3%7D>
</p>

where $v_{1,2,3}$ is the average postsynaptic membrane potential of populations of the excitatory stellate cells, inhibitory interneuron, and excitatory pyramidal cell populations, respectively.
The output $y(t) = v_1(t) - v_2(t)$ is the EEG signal.




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

![alt text](https://github.com/Davi1990/PyTepFit/tree/main/img/Figure_3.PNG)


Comparison between simulated and empirical TMS-EEG data in source space. A) TMS-EEG time series showing a robust recovery of grand-mean empirical TEP patterns in model-generated EEG time series. B) Source reconstructed TMS-evoked propagation pattern dynamics for empirical (top) and simulated (bottom) data. C) Time-wise permutation test results showing the significant Pearson correlation coefficient (tp) and the corresponding reversed p-values (bottom) for every vertex. Network-based dSPM values extracted for the grand mean empirical (left) and simulated (right) source-reconstructed time series.

![alt text](https://github.com/Davi1990/PyTepFit/tree/main/img/Figure_4.PNG)


|   | Run | View |
| - | --- | ---- |
| Params exploration | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Davi1990/PyTepFit/blob/master/nb/Params_exploration.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/Davi1990/PyTepFit/blob/master/nb/model_fit.ipynb?flush_cache=true) |


Synaptic time constant of inhibitory population affects early and late TEP amplitudes. A) Singular value decomposition (SVD) topoplots for simulated (top) and empirical (bottom) TMS-EEG data. By looking at simulated (top) and empirical (bottom) grand mean timeseries, results revealed that the first (orange outline) and the second (blue outline) components were located ∼65ms and ∼110ms after the TMS pulse. B) First (orange outline) and second (blue outline) components latency and amplitude were extracted for every subject and the distribution plots (top row) show the time location where higher cosine similarity with the SVD components were found. Scatter plots (bottom row) show a negative (left) and positive (right) significant correlation between the synaptic time constant of the inhibitory population and the amplitude of the the first and second component. C) Model-generated first and second SVD components for 2 representative subjects with high (top) and low (bottom) value for the synaptic time constant of the inhibitory population. Topoplots show that higher synaptic time constant affects the amplitude of the individual first and second SVD components. D) Model-generated TMS-EEG data were run using the optimal (top right) or a value decreased by 85% (central left) for the synaptic time constant of the inhibitory population. Absolute values for different values of the synaptic time constant of the inhibitory population (bottom right). Results show an increase in the amplitude of the early first local component and a decrease of the second latest global component

![alt text](https://github.com/Davi1990/PyTepFit/tree/main/img/Figure_6.PNG)
