# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 19:44:16 2018

@author: nrgurram
"""

import subprocess
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
        
dataset_paths = ["..\\dataset\\retail1\\retail1.txt", "..\\dataset\\retail2.txt"]
min_sup_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3];
#min_sup_vals = [3,4]
run_times = [];
algorithms = ["Apriori", "FPGrowth_itemsets", "Eclat"]
plot_figure = 0
pp = PdfPages("result.pdf")

def return_run_time(algo, input_file, min_sup):
    result = []
    command = "java -jar ..\\spmf.jar run " + algo + " " + input_file + " output.txt " + min_sup
    result = subprocess.check_output(command.split())
    matched = re.search(r"(\d+)\sms", str(result), re.MULTILINE)
    return int(matched.group(1))

for alg in algorithms:
    plot_figure += 1; 
    fig = plt.figure(plot_figure)
    plt.title(alg)
    plt.xlabel("Minium Support (in percentage)")
    plt.ylabel("Computation time")
    for path in dataset_paths:
        for min_sup in min_sup_vals:
            run_times.append(return_run_time(alg, path, str(min_sup) + "%"));
        plt.plot(min_sup_vals, run_times)        
        run_times = []
    pp.savefig();
plt.show()
pp.close()