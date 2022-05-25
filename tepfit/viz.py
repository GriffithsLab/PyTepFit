# Viz functions
"""
miscelallaneous functions and classes to visualize simulation data
"""


"""
Importage
"""

from scipy.io import loadmat
# Suppress warnings; keeps things cleaner
import warnings
warnings.filterwarnings('ignore')


# Standard scientific python import commands
import os,sys,glob,numpy as np,pandas as pd,seaborn as sns
sns.set_style('white')

#%matplotlib inline
from matplotlib import pyplot as plt
import pickle
from surfer import Brain
from mayavi import mlab
import pylab
import glob
import mne
import os.path as op
from scipy.stats import norm
from scipy import stats
import nibabel as nib
import scipy
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import os
from PIL import Image
from IPython import display



def plot_source_activity(data, fwd, noise_cov, inv, real_src_path, annot2use, subjects_dir, subject,
                        eeg_dir, method='dSPM', hemi='both', clim= 'auto',
                         time_viewer=True, views=['dorsal'],
                         surface='inflated',
                         alpha=0.5, size=(800, 400)):

    '''
    computing and plotting simulated time series using mne


    Parameters
    ----------
    data = np.array |
           simulated timeseries
           array (dim: number of regions X time)
    fwd = str |
          path to Forward solution
    noise_cov = str |
                path to noise_cov
    inv = str |
          path to Inverse solution
    real_src_path = str |
                    Path to a source-estimate file
    annot2use = str |
                name of the parcellation to use (e.g Schaefer2018_200Parcels_7Networks_order.annot)
    subjects_dir = str |
                  path to the subjects' folders
    subject = str |
              subject folder name that you want to plot
    eeg_dir = str |
            path to eegfile
    method (optional) = str |
            Use minimum norm (e.g. dSPM, sLORETA, or eLORETA)
            Default is dSPM
    hemi (optional)= str |
        (ie 'lh', 'rh', 'both', or 'split'). In the case
        of 'both', both hemispheres are shown in the same window.
        In the case of 'split' hemispheres are displayed side-by-side
        in different viewing panes
        Default is both
    clim (optional)= str | dict
         Colorbar properties specification. If ‘auto’, set clim automatically
         based on data percentiles. If dict, should contain:
            - kind‘value’ | ‘percent’
            Flag to specify type of limits.
            - limslist | np.ndarray | tuple of float, 3 elements
            Lower, middle, and upper bounds for colormap.
            - pos_limslist | np.ndarray | tuple of float, 3 elements
            Lower, middle, and upper bound for colormap. Positive values will be
            mirrored directly across zero during colormap construction to obtain
            negative control points.
    time_viewer (optional) = bool | str
                      Display time viewer GUI. Can be “auto”, which is True for
                      the PyVista backend and False otherwise.
                      Default is True
    views (optional) = str | list
                View to use. Can be any of:
                ['lateral', 'medial', 'rostral', 'caudal', 'dorsal', 'ventral',
                 'frontal', 'parietal', 'axial', 'sagittal', 'coronal']
                Three letter abbreviations (e.g., 'lat') are also supported.
                Using multiple views (list) is not supported for mpl backend.
                Default is 'dorsal'
    surface=  str |
        freesurfer surface mesh name (ie 'white', 'inflated', etc.).
        Default is inflated
    alpha (optional) = float |
        Alpha value to apply globally to the surface meshes. Defaults to 0.4.
        Default is 0.5
    size (optional) = float or tuple of float |
                      The size of the window, in pixels. can be one number to specify a
                      square window, or the (width, height) of a rectangular window.
                      Default is (800, 400)

    '''
    #Importing empirical source estimate
    stcs = mne.read_source_estimate(real_src_path)


    #Computing regmap for a specific parcellation
    lh_regmap = nib.freesurfer.io.read_annot(op.join(subjects_dir + \
                                    subject +  '/label/lh.' + annot2use))[0]
    rh_regmap = nib.freesurfer.io.read_annot(op.join(subjects_dir + \
                                        subject +  '/label/rh.' + annot2use))[0]
    rh_regmap_fixed = rh_regmap+np.max(lh_regmap)
    rh_regmap_fixed[rh_regmap_fixed == np.max(lh_regmap)] = 0
    lh_vtx2use= lh_regmap[stcs.vertices[0]]
    rh_vtx2use= rh_regmap_fixed[stcs.vertices[1]]
    regions_idx = np.concatenate([lh_vtx2use, rh_vtx2use])


    #Creating the leadfield matrix to use down the line
    leadfield = fwd['sol']['data']
    fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                             use_cps=True)
    leadfield = fwd_fixed['sol']['data']

    new_leadfield = []
    for xx in range(1,np.unique(regions_idx).shape[0]):
        new_leadfield.append(np.mean(np.squeeze(leadfield[:,np.where(regions_idx==xx)]),axis=1))

    new_leadfield = np.array(new_leadfield).T


    #Moving timeseries to EEG space using the leadfield matrix
    EEG = new_leadfield.dot(data).T # EEG shape [n_samples x n_eeg_channels]
    # reference is mean signal, tranposing because trailing dimension of arrays must agree
    EEG = (EEG.T - EEG.mean(axis=1).T).T
    EEG = EEG - EEG.mean(axis=0)  # center EEG
    simulated_EEG = EEG.T


    #Importing empirical EEG and creating an mne EEG object for simulated timeseries
    epochs = mne.read_epochs(eeg_dir)
    evoked = epochs.average()
    sim_EEG = evoked.copy()
    sim_EEG.data = simulated_EEG*1e-5
    sim_EEG.times = sim_EEG.times[:simulated_EEG.data.shape[1]]


    #Computing source for simulated timeseries
    inv_sim = mne.minimum_norm.make_inverse_operator(sim_EEG.info, forward=fwd, noise_cov=noise_cov)
    stc_sim = mne.minimum_norm.apply_inverse(sim_EEG, inv, method=method)


    #Plotting simulated source timeseries
    mne.viz.set_3d_backend("notebook")
    subjects_dir = subjects_dir
    subject = subject
    os.environ['SUBJECTS_DIR'] = subjects_dir
    brain = stc_sim.plot(subjects_dir=subjects_dir, subject=subject, clim=clim,
                         hemi=hemi, time_viewer=time_viewer, views=views,
                         surface=surface, alpha=alpha, size=size)

    return stc_sim, sim_EEG




def comparison_empirical_and_simulation(data, fwd, noise_cov, inv,
                                        real_src_path, annot2use,
                                        subjects_dir, subject,
                                        eeg_dir, time_range, space='source',
                                        metric2use= 'correlation',
                                        movie= False, windows=None,
                                        method='dSPM', opacity=0.5,
                                        colormap='coolwarm', views=['dorsal'],
                                        clim=(0, 0.02), distance=600.0,
                                        cortex='ivory'):


    '''
    computing and plotting comparison between empirical and simulated data
    both in the source and in the eeg space


    Parameters
    ----------
    data = np.array |
           simulated timeseries
           array (dim: number of regions X time)
    fwd = str |
          path to Forward solution
    noise_cov = str |
                path to noise_cov
    inv = str |
          path to Inverse solution
    real_src_path = str |
                    Path to a source-estimate file
    annot2use = str |
                name of the parcellation to use (e.g Schaefer2018_200Parcels_7Networks_order.annot)
    subjects_dir = str |
                  path to the subjects' folders
    subject = str |
              subject folder name that you want to plot
    eeg_dir = str |
            path to eegfile
    time_range: = float or tuple of float |
                      time interval for the timeseries comparison in sec
                      (e.g. -0.5, -0.1)
    space (optional) = str |
            space where the comparison analysis will be performed. Options are
            'source' or 'eeg'
            Default is 'source'
    metric2use (optional) = str |
            use any of the following:
                - correlation: I will compute the standard pearson coefficient
                - cosine: I will compute the cosine similarity
                - cross_correlation: I will compute the the cross correlation
            Default is 'correlation'
    movie (optional) = bool |
            if True, a dynamic comparison over time will be computed and
            export as gif. movie
            Default is False
    windows(optional) = int |
            only if movie is True. Specify the time resolution of the dynamic
            comparison
            Default is None
    method (optional) = str |
            Use minimum norm (e.g. dSPM, sLORETA, or eLORETA)
            Default is dSPM
    opacity (optional) = float in [0, 1] |
            Value to control opacity of the cortical surface
            Default is 0.5
    colormap (optional) =  str |
            matplotlib colormap to be used in the visualization. Options can be
            founded here: https://matplotlib.org/stable/tutorials/colors/colormaps.html
            Default is 'coolwarm'
    views (optional) = str | list
                View to use. Can be any of:
                ['lateral', 'medial', 'rostral', 'caudal', 'dorsal', 'ventral',
                 'frontal', 'parietal', 'axial', 'sagittal', 'coronal']
                Three letter abbreviations (e.g., 'lat') are also supported.
                Using multiple views (list) is not supported for mpl backend.
                Default is 'dorsal'
    clim (optional) = str | dict
         Colorbar properties specification with vmin and vmax values.
         Default is (0, 0.02)
    distance (optional): = float |
        Default is 600.0,
    cortex (optional) = str, tuple, dict, or None |
                Specifies how the cortical surface is rendered. Options:
                1) The name of one of the preset cortex styles: 'classic' (default),
                 'high_contrast', 'low_contrast', or 'bone'.
                2) A color-like argument to render the cortex as a single color,
                 e.g. 'red' or (0.1, 0.4, 1.). Setting this to None is equivalent
                 to (0.5, 0.5, 0.5).
                3) The name of a colormap used to render binarized curvature values, e.g., Grays.
                4) A list of colors used to render binarized curvature values.
                Only the first and last colors are used. E.g., [‘red’, ‘blue’] or
                [(1, 0, 0), (0, 0, 1)].
                5)A container with four entries for colormap (string specifiying
                the name of a colormap), vmin (float specifying the minimum value
                for the colormap), vmax (float specifying the maximum value for
                the colormap), and reverse (bool specifying whether the colormap
                should be reversed. E.g., ('Greys', -1, 2, False).
                6) A dict of keyword arguments that is passed on to the call to surface.
        Default is 'ivory'


    '''


    def NormalizeData(data):
         return (data - np.min(data)) / (np.max(data) - np.min(data))
    #Importing empirical source estimate

    stcs = mne.read_source_estimate(real_src_path)


    #Computing regmap for a specific parcellation
    lh_regmap = nib.freesurfer.io.read_annot(op.join(subjects_dir + \
                                    subject +  '/label/lh.' + annot2use))[0]
    rh_regmap = nib.freesurfer.io.read_annot(op.join(subjects_dir + \
                                        subject +  '/label/rh.' + annot2use))[0]
    rh_regmap_fixed = rh_regmap+np.max(lh_regmap)
    rh_regmap_fixed[rh_regmap_fixed == np.max(lh_regmap)] = 0
    lh_vtx2use= lh_regmap[stcs.vertices[0]]
    rh_vtx2use= rh_regmap_fixed[stcs.vertices[1]]
    regions_idx = np.concatenate([lh_vtx2use, rh_vtx2use])


    #Creating the leadfield matrix to use down the line
    leadfield = fwd['sol']['data']
    fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                             use_cps=True)
    leadfield = fwd_fixed['sol']['data']

    new_leadfield = []
    for xx in range(1,np.unique(regions_idx).shape[0]):
        new_leadfield.append(np.mean(np.squeeze(leadfield[:,np.where(regions_idx==xx)]),axis=1))

    new_leadfield = np.array(new_leadfield).T


    #Moving timeseries to EEG space using the leadfield matrix
    EEG = new_leadfield.dot(data).T # EEG shape [n_samples x n_eeg_channels]
    # reference is mean signal, tranposing because trailing dimension of arrays must agree
    EEG = (EEG.T - EEG.mean(axis=1).T).T
    EEG = EEG - EEG.mean(axis=0)  # center EEG
    simulated_EEG = EEG.T


    #Importing empirical EEG and creating an mne EEG object for simulated timeseries
    epochs = mne.read_epochs(eeg_dir)
    evoked = epochs.average()
    sim_EEG = evoked.copy()
    sim_EEG.data = simulated_EEG*1e-5
    sim_EEG.times = sim_EEG.times[:simulated_EEG.data.shape[1]]


    #Computing source for simulated timeseries
    inv_sim = mne.minimum_norm.make_inverse_operator(sim_EEG.info, forward=fwd, noise_cov=noise_cov)
    stc_sim = mne.minimum_norm.apply_inverse(sim_EEG, inv, method=method)

    if space=='source':
        #Comparing empirical data and simulated data in source space

        if movie:
            starting_pnt = (np.abs(epochs.times - (time_range[0]))).argmin()
            ending_pnt = (np.abs(epochs.times - (time_range[1]))).argmin()
            window_size = int((ending_pnt - starting_pnt) / windows)
            source_corr=np.zeros((stc_sim.data.shape[0], windows))
            source_similarity=np.zeros((stc_sim.data.shape[0], windows))
            source_cross=np.zeros((stc_sim.data.shape[0], windows))
            for tt in range(windows):
                X_simulated = np.abs(scipy.stats.zscore(stc_sim.data[:,starting_pnt:starting_pnt+window_size]))
                X_empirical = np.abs(scipy.stats.zscore(stcs.data[:,starting_pnt:starting_pnt+window_size]))

                for bb in range(X_simulated.shape[0]):
                    source_corr[bb,tt]= (np.corrcoef(X_simulated[bb], X_empirical[bb])[1][0])
                    source_similarity[bb,tt]=(cosine_similarity(X_simulated[bb].reshape(1, -1), X_empirical[bb].reshape(1, -1))[0][0])
                    source_cross[bb,tt]=np.mean(scipy.correlate(X_simulated[bb], X_empirical[bb], mode='same'))

                starting_pnt=starting_pnt+window_size

            source_cross = NormalizeData(source_cross)


            starting_pnt = (np.abs(epochs.times - (time_range[0]))).argmin()
            ending_pnt = (np.abs(epochs.times - (time_range[1]))).argmin()
            for cc in range(source_similarity.shape[1]):
                if metric2use=='correlation':
                    data2plot = source_corr[:,cc]
                elif metric2use=='cosine':
                    data2plot = source_similarity[:,cc]
                elif metric2use=='cross_correlation':
                    data2plot = source_cross[:,cc]

                #Left Hemiphere
                lh_correlation = np.zeros((lh_regmap.shape[0]))

                for xx in range(1,np.unique(lh_regmap).shape[0]):
                    lh_correlation[np.where(lh_regmap==xx)[0]] = data2plot[xx-1]

                #Right Hemiphere
                rh_correlation = np.zeros((rh_regmap.shape[0]))

                for xx in range(1,np.unique(rh_regmap).shape[0]):
                    rh_correlation[np.where(rh_regmap==xx)[0]] = data2plot[xx+100-1]

                brain = Brain(subject, subjects_dir=subjects_dir, hemi='both', surf='pial', cortex=cortex, alpha=opacity,
                              views=views)
                brain.add_data(lh_correlation,  min=clim[0], max=clim[1], hemi='lh', colormap=colormap)
                brain.add_data(rh_correlation,  min=clim[0], max=clim[1], hemi='rh', colormap=colormap)
                brain.add_text(x=0.1, y=0.9, text=str(epochs.times[starting_pnt]) + 'ms/' + str(epochs.times[starting_pnt+window_size]) + 'ms',
                               name=str(epochs.times[starting_pnt]) + 'ms/' + str(epochs.times[starting_pnt+window_size]) + 'ms',
                               font_size=20)

                starting_pnt=starting_pnt+window_size
                file = 'simulation_surface%03.0f.png' %cc
                mlab.view(distance=distance)
                mlab.savefig(file)
                mlab.close(all=True)

            #Making a gif
            fp_in = op.join(os.getcwd() + "/simulation_surface*")
            fp_out = op.join('source_' + metric2use + '.gif')
            img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
            img.save(fp=fp_out, format='GIF', append_images=imgs,
                     save_all=True, duration=200, loop=0)

            for filename in glob.glob("simulation_surface*"):
                os.remove(filename)

            source_comparison = {"cosine": source_similarity,
                                 "correlation": source_corr,
                                 "cross_correlation": source_cross}


            return source_comparison, display.Image(filename=fp_out,embed=True)

        else:
            starting_pnt = (np.abs(epochs.times - (time_range[0]))).argmin()
            ending_pnt = (np.abs(epochs.times - (time_range[1]))).argmin()
            X_simulated = np.abs(scipy.stats.zscore(stc_sim.data[:,starting_pnt:ending_pnt]))
            X_empirical = np.abs(scipy.stats.zscore(stcs.data[:,starting_pnt:ending_pnt]))

            source_corr=[]
            source_similarity=[]
            source_cross=[]

            for bb in range(X_simulated.shape[0]):
                source_corr.append(np.corrcoef(X_simulated[bb], X_empirical[bb])[1][0])
                source_similarity.append(cosine_similarity(X_simulated[bb].reshape(1, -1), X_empirical[bb].reshape(1, -1))[0][0])
                source_cross.append(scipy.correlate(X_simulated[bb], X_empirical[bb], mode='same'))

            source_corr = np.array(source_corr)
            source_similarity = np.array(source_similarity)
            source_cross = NormalizeData(np.array(np.mean(source_cross, axis=1)))

            if metric2use=='correlation':
                data2plot = source_corr
            elif metric2use=='cosine':
                data2plot = source_similarity
            elif metric2use=='cross_correlation':
                data2plot = source_cross

            #Left Hemiphere
            lh_correlation = np.zeros((lh_regmap.shape[0]))

            for xx in range(1,np.unique(lh_regmap).shape[0]):
                lh_correlation[np.where(lh_regmap==xx)[0]] = data2plot[xx-1]

            #Right Hemiphere
            rh_correlation = np.zeros((rh_regmap.shape[0]))

            for xx in range(1,np.unique(rh_regmap).shape[0]):
                rh_correlation[np.where(rh_regmap==xx)[0]] = data2plot[xx+100-1]

            brain = Brain(subject, subjects_dir=subjects_dir, hemi='both', surf='pial', cortex=cortex, alpha=opacity,
                          views=views)
            brain.add_data(lh_correlation,  min=clim[0], max=clim[1], hemi='lh', colormap=colormap)
            brain.add_data(rh_correlation,  min=clim[0], max=clim[1], hemi='rh', colormap=colormap)
            brain.add_text(x=0.1, y=0.9, text='average ' + str(time_range[0]) + 'ms/' + str(time_range[1]) + 'ms',
                           name='average ' + str(time_range[0]) + 'ms/' + str(time_range[1]) + 'ms',
                           font_size=20)

            mlab.view(distance=distance)
            mlab.show()

            return data2plot

    elif space=='eeg':

        if movie:
            starting_pnt = (np.abs(epochs.times - (time_range[0]))).argmin()
            ending_pnt = (np.abs(epochs.times - (time_range[1]))).argmin()
            window_size = int((ending_pnt - starting_pnt) / windows)
            source_corr=np.zeros((sim_EEG.data.shape[0], windows))
            eeg_similarity=np.zeros((sim_EEG.data.shape[0], windows))
            eeg_corr=np.zeros((sim_EEG.data.shape[0], windows))
            eeg_cross=np.zeros((sim_EEG.data.shape[0], windows))
            #Loop over windows number
            for tt in range(windows):
                X_simulated = np.abs(scipy.stats.zscore(sim_EEG.data[:,starting_pnt:starting_pnt+window_size]))
                X_empirical = np.abs(scipy.stats.zscore(evoked.data[:,starting_pnt:starting_pnt+window_size]))
                #Loop over electrodes
                for bb in range(X_simulated.shape[0]):
                    eeg_corr[bb,tt]= (np.corrcoef(X_simulated[bb], X_empirical[bb])[1][0])
                    eeg_similarity[bb,tt]=(cosine_similarity(X_simulated[bb].reshape(1, -1), X_empirical[bb].reshape(1, -1))[0][0])
                    eeg_cross[bb,tt]=np.mean(scipy.correlate(X_simulated[bb], X_empirical[bb], mode='same'))

                starting_pnt=starting_pnt+window_size

            source_cross = NormalizeData(eeg_cross)

            standard_1020_montage = mne.channels.make_standard_montage('standard_1020')

            ch_names=evoked.info['ch_names']
            ch_names_lower = np.array([x.lower() if isinstance(x, str) else x for x in ch_names])

            standard_1020_ch_names_lower = np.array([x.lower() if isinstance(x, str)\
                                                     else x for x in standard_1020_montage.ch_names]).tolist()

            ch_idx = []
            for xx in range(len(ch_names_lower)):
                ch_idx.append(np.where(np.array(ch_names_lower[xx])==np.array(standard_1020_ch_names_lower))[0][0])

            ch_idx = np.array(ch_idx)


            standard_1020_montage_ch_pos = []
            for xx in standard_1020_montage.get_positions()['ch_pos'].keys():
                idx_tmp = np.where(np.array(list(standard_1020_montage.get_positions()['ch_pos'].keys())) == xx)[0][0]
                standard_1020_montage_ch_pos.append(standard_1020_montage.get_positions()['ch_pos'][xx])

            standard_1020_montage_ch_pos = np.array(standard_1020_montage_ch_pos)

            standard_1020_montage_ch_pos_filtered = standard_1020_montage_ch_pos[ch_idx]*1000

            colormap=colormap
            color_map=plt.get_cmap(colormap)


            opacity=opacity


            surf='outer_skin_surface'
            hemi='rh'

            starting_pnt = (np.abs(epochs.times - (time_range[0]))).argmin()
            ending_pnt = (np.abs(epochs.times - (time_range[1]))).argmin()
            for cc in range(eeg_similarity.shape[1]):
                if metric2use=='correlation':
                    data2plot = eeg_corr[:,cc]
                elif metric2use=='cosine':
                    data2plot = eeg_similarity[:,cc]
                elif metric2use=='cross_correlation':
                    data2plot = eeg_cross[:,cc]

                coords = standard_1020_montage_ch_pos_filtered
                new_coords = coords.copy()
                new_scale_factor = data2plot.copy()

                brain = Brain(subject, subjects_dir=subjects_dir, hemi=hemi, surf=surf, views=views,
                              cortex=cortex, alpha=opacity)

                hemi = 'rh'
                for roi in range(new_coords.shape[0]):
                    rgba = color_map(data2plot[roi])
                    brain.add_foci(new_coords[roi], color=(rgba[0], rgba[1], rgba[2]), scale_factor=new_scale_factor[roi],hemi=hemi)

                brain.add_text(x=0.1, y=0.9, text=str(epochs.times[starting_pnt]) + 'ms/' + str(epochs.times[starting_pnt+window_size]) + 'ms',
                               name=str(epochs.times[starting_pnt]) + 'ms/' + str(epochs.times[starting_pnt+window_size]) + 'ms',
                               font_size=20)

                starting_pnt=starting_pnt+window_size
                file = 'simulation_surface%03.0f.png' %cc
                mlab.view(distance=distance)
                mlab.savefig(file)
                mlab.close(all=True)

            #Making a gif
            fp_in = op.join(os.getcwd() + "/simulation_surface*")
            fp_out = op.join('eeg_' + metric2use + '.gif')
            img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
            img.save(fp=fp_out, format='GIF', append_images=imgs,
                     save_all=True, duration=200, loop=0)

            for filename in glob.glob("simulation_surface*"):
                os.remove(filename)

            eeg_comparison = {"cosine": eeg_similarity,
                                 "correlation": eeg_corr,
                                 "cross_correlation": eeg_cross}


            return eeg_comparison, display.Image(filename=fp_out,embed=True)



        else:

            #Comparing empirical data and simulated data in eeg space
            starting_pnt = (np.abs(epochs.times - (time_range[0]))).argmin()
            ending_pnt = (np.abs(epochs.times - (time_range[1]))).argmin()

            X_simulated = sim_EEG.data[:,starting_pnt:ending_pnt]
            Y_empirical = evoked.data[:,starting_pnt:ending_pnt]

            eeg_corr=[]
            eeg_similarity=[]
            eeg_cross=[]

            for bb in range(X_simulated.shape[0]):
                eeg_corr.append(np.corrcoef(X_simulated[bb], Y_empirical[bb])[1][0])
                eeg_similarity.append(cosine_similarity(X_simulated[bb].reshape(1, -1), Y_empirical[bb].reshape(1, -1))[0][0])
                eeg_cross.append(scipy.correlate(X_simulated[bb], Y_empirical[bb], mode='same'))

            eeg_corr = np.array(eeg_corr)
            eeg_similarity = np.array(eeg_similarity)
            eeg_cross = np.array(eeg_cross)

            standard_1020_montage = mne.channels.make_standard_montage('standard_1020')

            ch_names=evoked.info['ch_names']
            ch_names_lower = np.array([x.lower() if isinstance(x, str) else x for x in ch_names])

            standard_1020_ch_names_lower = np.array([x.lower() if isinstance(x, str)\
                                                     else x for x in standard_1020_montage.ch_names]).tolist()

            ch_idx = []
            for xx in range(len(ch_names_lower)):
                ch_idx.append(np.where(np.array(ch_names_lower[xx])==np.array(standard_1020_ch_names_lower))[0][0])

            ch_idx = np.array(ch_idx)


            standard_1020_montage_ch_pos = []
            for xx in standard_1020_montage.get_positions()['ch_pos'].keys():
                idx_tmp = np.where(np.array(list(standard_1020_montage.get_positions()['ch_pos'].keys())) == xx)[0][0]
                standard_1020_montage_ch_pos.append(standard_1020_montage.get_positions()['ch_pos'][xx])

            standard_1020_montage_ch_pos = np.array(standard_1020_montage_ch_pos)

            standard_1020_montage_ch_pos_filtered = standard_1020_montage_ch_pos[ch_idx]*1000



            colormap=colormap
            color_map=plt.get_cmap(colormap)


            opacity=opacity


            surf='outer_skin_surface'
            hemi='rh'


            if metric2use=='correlation':
                data2plot = NormalizeData(eeg_corr)
            elif metric2use=='cosine':
                data2plot =  NormalizeData(eeg_similarity)
            elif metric2use=='cross_correlation':
                data2plot = eeg_cross

            coords = standard_1020_montage_ch_pos_filtered
            new_coords = coords.copy()
            new_scale_factor = data2plot.copy()

            brain = Brain(subject, subjects_dir=subjects_dir, hemi=hemi, surf=surf, views=views,
                          cortex=cortex, alpha=opacity)

            hemi = 'rh'
            for roi in range(new_coords.shape[0]):
                rgba = color_map(data2plot[roi])
                brain.add_foci(new_coords[roi], color=(rgba[0], rgba[1], rgba[2]), scale_factor=new_scale_factor[roi],hemi=hemi)

            brain.add_text(x=0.1, y=0.9, text='average ' + str(time_range[0]) + 'ms/' + str(time_range[1]) + 'ms',
                           name='average ' + str(time_range[0]) + 'ms/' + str(time_range[1]) + 'ms',
                           font_size=20)
            mlab.view(distance=distance)
            mlab.show()

            return data2plot
