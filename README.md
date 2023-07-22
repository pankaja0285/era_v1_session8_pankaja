Purpose: Apply 3 different Normalization methods.
Based on CiFAR 10 dataset
Create 3 different one for each Normalization methods:-
Batch Normalization
Group Normalization
Linear Normalization
Project Setup:
Clone the project as shown below:-

$ git clone git@github.com:pankaja0285/era_v1_session8_pankaja.git
$ cd era_v1_session8_pankaja
About the file structure
|__config __config.yaml
|__data |__data_analysis |__data_loader __load_data.py
__albumentation.py
|__models __model.py
|__utils __dataset.py<br/ __engine.py
__helper.py
__plot_metrics.py
__test.py
__train.py
|__S8.ipynb
|__README.nd

NOTE: List of libraries required: torch and torchsummary, tqdm for progress bar, which are installed using requirements.txt

One of 2 ways to run any of the notebooks, for instance S8.ipynb notebook:

Using Anaconda prompt - Run as an administrator start jupyter notebook from the folder era_v1_session8_pankaja and run it off of your localhost
NOTE: Without Admin privileges, the installs will not be correct and further import libraries will fail.
jupyter notebook
Upload the notebook folder era_v1_session8_pankaja to google colab at colab.google.com and run it on colab
Context: The drill down analysis starts from Step 0 through Step 6 and the analysis is laid out as Target, Results and Analysis.

Target: what is the target we are setting up as
Results: what results we are getting
Analysis: basically analyzing what we are doing in the model in this step, what is the algorithm etc.,
Batch Normalization:
File used: models/model.py, model with Net1 Class

Target: - create a model with Batch Normalization as the normalization method

Results:

Total parameters: 52,576
Train accuracy of and test accuracy of 77.69
Analysis:

To see how the accuracy is using Batch Normalization method.
Group Normalization:
File used: models/model.py, model with Net2 Class

Target: - create a model with Group Normalization as the normalization method

Results:

Total parameters: 52,576
Train accuracy of and test accuracy of 77.69
Analysis:

To see how the accuracy is using Group Normalization method.
Linear Normalization:
File used: models/model.py, model with Net3 Class

Target: - create a model with Linear Normalization as the normalization method

Results:

Total parameters: 52,576
Train accuracy of and test accuracy of 77.69
Analysis:

To see how the accuracy is using Linear Normalization method.
Contributing:
For any questions, bug(even typos) and/or features requests do not hesitate to contact me or open an issue!
