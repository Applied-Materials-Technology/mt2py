import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path

def noise_assess(folder_path,output_average_path=None,plots=False):
    
    parent =folder_path
    files = list(parent.glob('*.tiff'))
    def get_ind(f):
        return int(f.stem.split('_')[1])
    files_sort = sorted(files,key=get_ind)


    base_image = np.array(Image.open(files_sort[0]),dtype=int)
    avg_image = np.empty((base_image.size))

    noise = np.empty((base_image.size,len(files_sort)-1))


    for i,file in enumerate(files_sort[1:]):
        current_image = np.array(Image.open(file),dtype=int)
        noise[:,i] = (current_image-base_image).ravel()
        avg_image += current_image.ravel()

    avg_image = avg_image/len(files_sort)

    if output_average_path is not None:
        out_avg = Image.fromarray(np.uint8(avg_image.reshape(base_image.shape)))
        out_avg.save(output_average_path)

    deviations = np.zeros(256)
    noise_deviation = np.nanstd(noise,axis=1)
    for i in range(256):
        inds = np.where(base_image.ravel()==i)[0]
        deviations[i] = np.nanmean(noise_deviation[inds])


    grey_levels = np.arange(256)

    mask = ~np.isnan(deviations)

    grey_levels = grey_levels[mask]
    deviations = deviations[mask]

    p =np.polynomial.polynomial.Polynomial.fit(grey_levels,deviations,6)
    
    if plots:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(grey_levels,deviations,color='k',label='Standard Deviation')
        ax.plot(p(grey_levels),color='crimson',label='6th Order Polynomial Fit')
        ax.legend()
        ax.set_xlabel('Grey Level')
        ax.set_ylabel('Noise Standard Deviation [Grey Levels]')
    return grey_levels, deviations, p