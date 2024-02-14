### plot_co_occurrences(sonar=son)
### plot_co_occurrences(sonar=son, labels=['acinar_s','alpha','beta'])
### ploc_co_occurrences(sonar=son, color_label='colors') <- column name in pandas data frame that stores colors


"""The following function is the large plotting function. 
Editional notes: 
* I really think that significance test should be included in the sonar object. Then we can reduce the number of arguments in the function.
    so all arguments like "significant_enrichment, significant_depletion, normalized_coocurrence"can go away, once they are in sonar
* I was a bit confused on how to make plots flexible. Maybe you should rewrite some functions with the gridspec package so that plots can be 
    flexible to take up to 30 cell types and adjust automatically to the number of cell-types.
"""

from datetime import datetime
import random
from distinctipy import distinctipy
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plotting_enrichment_report(significant_enrichment, significant_depletion, normalized_coocurrence, son, curves_colors = True, 
                               report_filename = "pancreas", file_extension=None, threshold_enrichment=2, threshold_depletion=2, seed_= 13): 
    """generates and saves a spatial correlation plot report.
    Generates a spatial correlation plot report and save is in the selected format, default=PDF.

    Args:
        significant_enrichment (numpy.ndarray): Matrix of enriched values.
        significant_depletion (numpy.ndarray): Matrix of depleted values.
        normalized_coocurrence (numpy.ndarray): Normalized co-occurrence data.
        son: Sonar object.
        curves_colors (bool, optional): Whether to generate distinct curve colors. Default is True.
        report_filename (str, optional): Base filename for the generated report. Default is "pancreas".
        file_extension (str, optional): File extension for the report file. Default is ".pdf".
        threshold_enrichment (float, optional): Threshold for enriched values. Default is 2.
        threshold_depletion (float, optional): Threshold for depleted values. Default is 2.
        seed_ (int, optional): Seed for random color generation. Default is 13.

    Outputs:
        None

    Generates a PDF report containing spatial correlation plots and heatmaps.
    """
    # The part which is responcible for creating the color maps and color bars, preprocessing the data:

    # defining variables for the enriched and depleted plot:
    significance_type_ = (significant_enrichment, significant_depletion)
    cmap_names_ = ['enr_cmap', 'depl_cmap']
    cmap_threshold_ = (threshold_enrichment, threshold_depletion)
    enrichment_color = [(0, 0, 0), (1, 0, 0)]
    depletion_color = [ (0, 0, 0),(0.4, 0.7, 0.9)]
    cmap_colors_ = (enrichment_color, depletion_color)

    # lists for storage values
    signif_type_result = []
    cmap_result = []
    # colormaps and colorbars for enriched and depleted:
    for signif_type, cmap_names, cmap_threshold, cmap_colors in zip(significance_type_, cmap_names_, cmap_threshold_, cmap_colors_):
        heatmap_modif = signif_type.copy()
        if cmap_threshold:
            heatmap_modif[heatmap_modif<cmap_threshold] = 0
        n_bins = 600  # Number of bins in the colormap. I used this number because the colorbar scale ranges from 0 to 6
        cmap = LinearSegmentedColormap.from_list(cmap_names, cmap_colors, N=n_bins)
        cmap_np = cmap(np.linspace(0, 1, 600))
        black = np.array([0, 0, 0, 1])
        cmap_np[:200, :] = black
        cmap_2black = ListedColormap(cmap_np)

        signif_type_result.append(heatmap_modif)
        cmap_result.append(cmap_2black)

    # Merging significances to create a common heatmap
    enr_val_modif = signif_type_result[0]
    depl_val_modif = signif_type_result[1]
    enr_depl_merged = enr_val_modif - depl_val_modif # Subtracting because results are reversed

    # colormaps and colorbars for merged:
    colors = [(0.4, 0.7, 0.9), (0, 0, 0), (1, 0, 0)]  # Blue, Black, Red
    n_bins = 1200  # Number of bins in the colormap
    cmap_name = 'cmap4merged'
    cmap4merged = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    cmap4merged_np = cmap4merged(np.linspace(0, 1, 1200))
    black = np.array([0, 0, 0, 1])
    cmap4merged_np[400:800, :] = black
    cmap4merged2black = ListedColormap(cmap4merged_np)



    # The part which is responcible for actual plotting and creating the report:

    # filename
    formatted_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") # just to make names of the output reports unique
    if file_extension is not None:
        filename = report_filename + formatted_datetime + file_extension
        pdf_pages = PdfPages(filename) # "open"the .pdf

    # colors
    if curves_colors:
        random.seed(seed_)
        N = 32# number of colours to generate
        curves_colors = distinctipy.get_colors(N) # generate N visually distinct colours


    for pivot_cell_type in range(normalized_coocurrence.shape[0]):
        # hidden heatmaps for colorbars
        fig_not_show, (ax_noshow_enri, ax_noshow_depl) = plt.subplots(ncols=2, figsize = (3,2))
        sns.heatmap(enr_val_modif[pivot_cell_type,:,:], cmap=cmap_result[0], ax=ax_noshow_enri, cbar=False, xticklabels=False, yticklabels=False, vmin = 0, vmax=6)
        sns.heatmap(-depl_val_modif[pivot_cell_type,:,:], cmap=cmap_result[1], ax=ax_noshow_depl, cbar=False, xticklabels=False, yticklabels=False, vmin = 0, vmax=6)
        plt.close()

        figwidth = 10 # Maybe should be in the function arguments
        figheight = 10
        fig, (ax_curve, ax_show_comb) = plt.subplots(2,1, figsize = (figwidth,figheight), sharex=True)
        plt.suptitle(f"Spatial correlation plot for {son.meta.index[pivot_cell_type]} cells", fontsize = 18)

        # Adjust the spacing between subplots
        plt.subplots_adjust(hspace=0.07)

        # Generating curves
        for target_cell_type in range(normalized_coocurrence.shape[0]):
            ax_curve.plot(normalized_coocurrence[pivot_cell_type,target_cell_type], color=curves_colors[target_cell_type % len(curves_colors)], 
            label=(son.meta.index.to_list()[target_cell_type] + " (" + str(son.meta["pixel_percentage"].to_list()[target_cell_type]) + "%)"))
            ax_curve.legend(loc="upper right", bbox_to_anchor=(1.4, 1), borderaxespad=1.5)
            ax_curve.set_ylim(0,5)
        ax_curve.set_ylabel("Sonar spatial enrichment")
        # ax_curve.annotate("Spatial correlation plot for ... cells", xy=(0.5,(np.max(ax_curve.get_ylim()))), fontsize = 18) # this can be used as a plot title

        # significance heatmap
        sns.heatmap(enr_depl_merged[pivot_cell_type,:,:], cmap=cmap4merged2black, ax=ax_show_comb, cbar=False, yticklabels = son.meta.index.to_list())
        ax_show_comb.set_xlabel('Distance from center (um)', fontsize = 12)
        ax_show_comb.text((normalized_coocurrence.shape[2]+2*(normalized_coocurrence.shape[2]/figwidth)),1,"Significance \n (-log10 p-value)", fontsize=12, ha="right", va='top')

        cbar_ax1 = fig.add_axes([9.3/figwidth, 0.12, 0.02, 0.25]) #parameters for colorbars (left, bottom, width, hight)
        cbar_ax1.text(-8.5/figwidth,35/figheight,"Enrichment", fontsize = 10, rotation=90) # current proportions are not optimal. Needs a better way. Maybe a gridspec.
        cbar_ax2 = fig.add_axes([10/figwidth, 0.12, 0.02, 0.25]) 
        cbar_ax2.text(-11.5/figwidth,38/figheight,"Depletion", fontsize = 10, rotation=90) 

        cbar_enr = fig.colorbar(ax_noshow_enri.collections[0], cax=cbar_ax1, location="right", use_gridspec=False, shrink=5, pad = 0.5)
        cbar_depl = fig.colorbar(ax_noshow_depl.collections[0], cax=cbar_ax2, location="right", use_gridspec=False, shrink=5)

        cbar_adapted_ticks = [0,0,2,0,4,0,6]
        cbar_enr.set_ticks(cbar_adapted_ticks)
        cbar_depl.set_ticks(cbar_adapted_ticks)

        # plt.savefig("output.png") # .pdf takes ages to generate, so I switched to .png

        # Save the current figure to the PDF file
        if file_extension is not None:
            pdf_pages.savefig()

        # Close the current figure
        plt.close()
                
    # Close the PDF file
    if file_extension is not None:
        pdf_pages.close()   
