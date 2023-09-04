### plot_co_occurrences(sonar=son)
### plot_co_occurrences(sonar=son, labels=['acinar_s','alpha','beta'])
### ploc_co_occurrences(sonar=son, color_label='colors') <- column name in pandas data frame that stores colors

# def plot(son):
#     co_occurrences = son.co_occurrences
#     celltypes = son.metadata.index
#     pixel_counts = son.pixel_counts

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn_image as isns
import seaborn as sns
from distinctipy import distinctipy
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
import random


def plot_enrichment_report(co_occurrence, significant_enrichment, metadata, 
                           threshold=0.0005, scaling = 1, #if the initial file was scaled
                           filename = "mouse_brain_", #if needed
                           seed_ = 13 #if needed
                           ):   
    random.seed(seed_)

    ct = list(metadata[metadata['pixel_proportions']>threshold].index)

    metadata["order_of_layers"] = list(range(len(metadata)))
        # metadata_reset = metadata.reset_index()
        # metadata_reset.rename(columns={'index': 'cell_type'}, inplace=True)
        # metadata_reset.set_index('order_of_layers', inplace=True)

    all_cell_types = list(metadata.index)
    cell_types_above_threshold = list(metadata.loc[metadata['pixel_proportions']>threshold, 'order_of_layers'])
    thresholded_cell_types = list(metadata.loc[metadata['pixel_proportions']>threshold].index)

    # colors    
    random.seed(seed_)
    # number of colours to generate
    N = len(cell_types_above_threshold)
    # generate N visually distinct colours
    colors = distinctipy.get_colors(N)


    # Create a PDF file
    pdf_filename = filename + ".pdf"
    pdf_pages = PdfPages(pdf_filename)


    for pivot_cell_type in cell_types_above_threshold:

        # Initiating a figure
        # fig, axes = plt.subplots(3,1, figsize=(10,len(cell_types_above_threshold)//2), sharex=True)
        fig = plt.figure(figsize=(14, 12))  # Overall figure size
        gs = gridspec.GridSpec(3, 1, height_ratios=[1,3,3])


            # Sonar cross-correlation curves
        ax_curves = plt.subplot(gs[0])
        for target_cell_type in cell_types_above_threshold:

                # y_maximum = normalized_coocur[pivot_cell_type,:,10:].mean() + normalized_coocur[pivot_cell_type,:,10:].std()*2.5

            ax_curves.plot(co_occurrence[pivot_cell_type,target_cell_type], label = all_cell_types[target_cell_type], color=colors[target_cell_type % len(colors)])
            ax_curves.legend(loc="upper right", bbox_to_anchor=(-0.2, 1))
            ax_curves.set_ylim(-0.5,5)
        ax_curves.set_ylabel('Relative enrichment', labelpad=15)

        # Define axes for each subplot
        heatmaps = [1, 2]

        # Define alternative types
        alternatives = significant_enrichment
        alt_name = ["less", "greater"]

        for heatmap_loc, alternative, a_name in zip(heatmaps, alternatives, alt_name):
            ax = plt.subplot(gs[heatmap_loc], sharex=ax_curves)
            # Significance heatmap of one-sided t-test
            sns.heatmap(-np.log10(alternative[pivot_cell_type, cell_types_above_threshold, :]), cmap="rocket", yticklabels=thresholded_cell_types, vmin=1.6, vmax=6, ax=ax, cbar=False)
            
            # Add a text annotation on the right side of the plot
            # ax.annotate(f"alternative={a_name}", xy=(0, 0), xytext=(620, 6), color="black",
            #                 rotation=-90, ha='right', va='center', fontsize=14)
            
            # Adjust the frame of the heatmap
            for spine in ['top', 'right', 'bottom', 'left']:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_linewidth(1)

            
        # Customize x-axis ticks
        xlox = np.linspace(0,co_occurrence.shape[2],co_occurrence.shape[2]//10) #use the shape of provided tensor!
        xtix = [str(int(i)) for i in xlox*scaling] 
        plt.xticks(xlox,xtix)   
        ax_curves.set_xticklabels(ax_curves.get_xticklabels(), rotation=90, ha='right')

        plt.tight_layout()

        # Create a color bar
        cbar_ax = fig.add_axes([0.91, 0.05, 0.02, 0.6])
        cbar = plt.colorbar(ax.collections[0], cax=cbar_ax)
        cbar.set_label('-Log10 Significance', rotation=90, labelpad=10)
        cbar.outline.set_linewidth(1)

        # Adjust the linewidth of legend lines
        legend_lines = ax_curves.get_legend().get_lines()
        for line in legend_lines:
            line.set_linewidth(5.0)

        plt.xlabel('Distance, um/px', labelpad=40) # Add x-axis label to the entire figure
        plt.suptitle(f"Cross-correlation of {all_cell_types[pivot_cell_type]}", fontsize=15, y=1.01)

        # Save the current figure to the PDF file
        pdf_pages.savefig()

        # Close the current figure
        plt.close()
            
    # # Close the PDF file
    pdf_pages.close()