# Bayesian Optimization via Tempered Posterior

This repo contains the official Python implementation of our proposed tempered Bayesian optimization framework described in the paper **: (placeholder). The current codebase implements tempered (or regular) GP surrogate, with generalized expected improvement acquization functions (including both probability of improvement, expected improvement, and more). Our tempering methodlogy can be generalized to other acquization functions as well.



# Repo Directories

To import the source code in ```src/```, do

```bash
python setup.py install
pip install -e .
```
Then in your python script editor:
```python
import bayesian_optimization as bo
``` 

The following directories are especially helpful:

- ```src/bayesian_optimization/BayesianOptimizer.py```: contains implementation of the main source code.
- ```project/x01_bo_experiments```: driver code for the simulation exercise in Section 6.1 of the paper.
- ```markdowns/10-25-25-experiments/result_analysis.ipynb```: result visualizations + figures/tables replications for Section 6.1 of the paper.
- ```project/x03_real_data```: driver code for the real voltage optimization experiment in Section 6.2 of the paper.
- ```markdowns/10-30-voltage-data/result_analysis.ipynb```: result visualizations + figures/tables replications for Section 6.2 of the paper.


# Experiment Replications


1. **Section 6.1 (simulation)**: Navigate to the ```project/x01_bo_experiments``` folder: we divide all $61$ functions into three batches and run BO in parallel for computational convieniece.

   - Run script ```s01_BO_experiments.py``` to generate simulation result for the first batch of functions.
   - Run script ```s02_BO_experiments_batch2.py``` to generate simulation result for the second batch of functions.
   - Run script ```s03_BO_experiments_batch3.py``` to generate simulation result for the thrid batch of functions.
   - There will be 3 separate pickle files in ```models/X01````
   - To visualize these results and replicate all the tables and figures in the paper , navigate to ```markdowns/10-25-25-experiments/result_analysis.ipynb``` and run all the cells.

2. **Section 6.2 (Voltage Optimization)**: Navigate to the ```project/x03_real_data``` folder:

   - Run script ```s01_BO_experiments.py``` to run BO for the voltage optimization data. The real data is included in the ```data/``` folder.
   - To visualize the results and replicate all the tables and figures in the paper , navigate to ```markdowns/10-30-voltage-data/result_analysis.ipynb``` and run all the cells.

  