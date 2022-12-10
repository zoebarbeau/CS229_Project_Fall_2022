Outline of folders:
- data: contains all data used for this project (not submitted because of size)
	- within this folder, there exists LES / RANS data. 
		- LES = very detailed simulation --> labels
		- RANS = coarse simulation --> data from which features are computed
- feature analysis: contains the feature analysis done on the mean, variance, max, min for all features
	- omitted since not part of the code
- features: contains functions used to pre-process the data
	- this includes conversion codes to change data formats and several computational codes to compute
	   features and labels for all cases.
- tke analysis: things related to creating the sample weighting scheme
	- omitted since not part of the code
- ML_models: this is where the runs are saved. 
	- for each run there are subfolders for each case, feature set, and normalization approach
	- these subfolders include weights at different epochs, loss, loss plot, error analysis plots
	- in the main directory for each run there are also summary analysis plots
	- we have included some sample runs with just FS4 & norm 1 (we did not include any RF ones 
          because those had RF models saved that were several hundred Mbs. Including more FS and norms
          is more MB so we are just giving one FS/norm.)
- utilities: contains codes that support loading, storing, and converting data
	- some of these codes are copied and placed in folder features/convert
	- there are some duplicate codes with slight changes to adjust for different types of runs
		- like if the ML models had sequtional inputs or a ton of different flow types
- Code history: all ML codes that we ran for this class 
	- some examples are left in the main folder for the ta to look at
- codes in main folder:
(Please note that these are listed in alphabetical order and not all codes are currently used)
	1. create_dictionary (2 versions) --> rather than computing the features for each trail run
	   these preprocessing codes compute all the features, and we save them into dictionaries to
	   later load at run time. This is no longer used. A utility function exists 
	2. create_grid_images: used to create .pngs of the grids of different flow types in the dataset
		- used strictly for report images
	3. plotting old results: used to create updated plots of past runs 
	4. print errors to copy: used to output pickle files into text that we could copy into excel
	5. TrainTest_Base: a general format on how  we ran the ML Alg. All TrainTest files are like the 
           base, but NN, RF, and kmeans variations
(^^^ TrainTest_Base is what TA should grade)
