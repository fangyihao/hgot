'''
Created on Jan. 18, 2024

@author: Yihao Fang
'''

import numpy as np
from scipy.stats import t, ttest_ind, ttest_rel

Vanilla_LM_EM_score =   [1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
DSP_EM_score =          [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
print("Vanilla_LM_EM_score:", np.mean(Vanilla_LM_EM_score))
print("DSP_EM_score:", np.mean(DSP_EM_score))

print("ttest_paired_two-sided:", ttest_rel(DSP_EM_score, Vanilla_LM_EM_score, alternative='two-sided'))
print("ttest_paired_one-sided:", ttest_rel(DSP_EM_score, Vanilla_LM_EM_score, alternative='greater'))




'''
#Compute the difference between the results
diff = [y - x for y, x in zip(DSP_score, HGOT_score)]
#Comopute the mean of differences
d_bar = np.mean(diff)
#compute the variance of differences
#sigma2 = np.var(diff)
sigma = np.std(diff)
#compute the total number of data points
n = len(HGOT_score)
#compute the modified variance
#sigma2_mod = sigma2 * (1/n)
#compute the t_static
#t_static =  d_bar / np.sqrt(sigma2_mod)
t_static =  d_bar / sigma*np.sqrt(1/n)
print("t:", t_static)


#Compute p-value and plot the results 
p_value = ((1 - t.cdf(t_static, n-1)))
print("p value:", p_value)
'''
#print("ttest_independent_two-sided:", ttest_ind(HGOT_score, DSP_score, alternative='two-sided'))
#print("ttest_independent_one-sided:", ttest_ind(HGOT_score, DSP_score, alternative='greater'))


