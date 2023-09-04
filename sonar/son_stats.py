import torch as t
import numpy as np
from scipy import stats
import statsmodels.stats.multitest as smm
import math

def significance_test(co_occurrence):
    """
    Perform significance tests on normalized co-occurrence data.

    This function conducts significance tests to determine if the normalized co-occurrence values
    are significantly different from the overall mean. It uses t-tests for each cell type pair and distance.

    Args:
        co_occurrence (numpy.ndarray): 3D array of normalized co-occurrence values.
    
    Returns:
        less_greater (list of numpy.ndarray): List containing 3D arrays of corrected p-values for "less" and "greater" alternatives. 
        "less" shows enrichment of a given cell-type on a certain area, while "greater"represent depletion.

    Note:
        The p-values are corrected for multiple hypothesis testing using the Benjamini-Hochberg procedure (method='fdr_bh').
        This correction helps control the False Discovery Rate (FDR) when conducting numerous tests.
        Corrected p-values are provided in the returned 3D arrays for further interpretation and analysis.
    """
    less_greater = [] # maybe should be changed to numpy.ndarray, but I find a list more convenient in this case

    # performing 1-sided ttest with two alternative options: lees and greater, to see both "ups" and "donws"
    alternatives = ["less", "greater"]
    for one_side in alternatives:

        significance_test = np.zeros((co_occurrence.shape[0],co_occurrence.shape[1],co_occurrence.shape[2]))
        for pivot_cell_type in range(co_occurrence.shape[0]):
            for target_cell_type in range(co_occurrence.shape[1]):
                for distance in range(co_occurrence.shape[2]):

                    value_to_test = co_occurrence[pivot_cell_type, target_cell_type, distance]
                    t_statistic, p_value = stats.ttest_1samp(co_occurrence[pivot_cell_type,:,distance], value_to_test, alternative=one_side)
                    significance_test[pivot_cell_type, target_cell_type, distance] = p_value

    # correction for multiple hypothesis testing
        significance_test_flatten = significance_test.flatten()
        corrected_p_values = smm.multipletests(significance_test_flatten, method='fdr_bh')[1] 
        corrected_p_values3D = corrected_p_values.reshape(co_occurrence.shape[0],co_occurrence.shape[1],co_occurrence.shape[2])
        
        less_greater.append(corrected_p_values3D)

    return less_greater