# **Purchase Prediction**

## Introduction
Many retailers are trapped in the death spiral of never-ending sales designed to “buy” customers with deals. This vicious cycle of price slashing is not sustainable. Retail profit margins are razor thin as it is. More worryingly, as much as five percent of their customer base typically is loss-making because these customers are so adept at using discounts, promotions and returning items. To overcome this issue, Machine Learning (ML) is utilized to understand and optimize complex, multi-touch customer journeys, products and services. Core customer-centric use cases should enable purchase prediction modeling.

This example creates an end-to-end Machine Learning solution to build a deeper understanding of customer segments based on historical purchases, which can be further used to implement operational and product marketing strategies. It also describes how we can leverage the [Intel® Distribution for Python*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html#gs.52te4z) and [Intel® Extension for Scikit-Learn*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/scikit-learn.html) to accelerate the pipeline. This distribution and extension are part of the [Intel® oneAPI AI Analytics Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html), which gives data scientists, AI developers, and researchers familiar Python tools and frameworks to accelerate end-to-end data science and analytics pipelines on Intel® architectures. Check out more workflow examples in the [Developer Catalog](https://developer.intel.com/aireferenceimplementations).

## Solution Technical Overview
One important aspect of purchase prediction for a new customer is to map them to a segment with similar customer purchase history. In this reference kit, we build a Machine Learning model based on the historic details of various customer purchases on different time frames to predict if a customer belongs to a certain segment. This would help the retail business to devise their operational and product marketing strategies based on real data when a customer comes back on the channel next time. 

One of the primary methods for deriving an understanding of how we can accomplish this is by analyzing and exploring different AI based Machine Learning algorithms on various feature sets to categorize the customers to appropriate segments.

The reference kit explores ways to have:
- Faster model development that helps in building purchase prediction models
- Performance efficient purchase prediction mechanisms

This Reference Solution approach involves clustering of data in the initial analysis to create the product categories, unsupervised Machine Learning technique (K-Means) is used for this. We also need to create and predict the customer segment based on their purchase which is solved as a classification problem. A K-Means model is used again for the creation of customer segments. The benchmarking is done only for the classifier models involved in predicting the customer segment with the features of known product categories from their product purchase history. For this, multiple below classifiers algorithms are analyzed:
- K-Nearest Neighbor
- Decision Tree
- Random Forest

![image](assets/e2e_flow_optimized.png)

The use case extends to demonstrate the advantages of using the Intel® oneAPI AI Analytics Toolkit on the task of building a targeted understanding of customer characteristics from purchase data. The savings gained from using the Intel® Extension for Scikit-learn can lead an analyst to more efficiently explore and understand customer archetypes, leading to better and more precise targeted solutions.

The solution contained in this repo uses the following Intel® packages:

* ***Intel® Distribution for Python\****

    The [Intel® Distribution for Python*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html#gs.52te4z) provides:
    * Scalable performance using all available CPU cores on laptops, desktops, and powerful servers
    * Support for the latest CPU instructions
    * Near-native performance through acceleration of core numerical and machine learning packages with libraries like the Intel® oneAPI Math Kernel Library (oneMKL) and Intel® oneAPI Data Analytics Library
    * Productivity tools for compiling Python code into optimized instructions
    * Essential Python bindings for easing integration of Intel® native tools with your Python* project

* ***Intel® Extension for Scikit-learn\****

    Designed for data scientists, [Intel® Extension for Scikit-Learn*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/scikit-learn.html) is a seamless way to speed up your Scikit-learn applications for machine learning to solve real-world problems This extension package dynamically patches scikit-learn estimators to use Intel® oneAPI Data Analytics Library (oneDAL) as the underlying solver, while achieving the speed up for your machine learning algorithms out-of-box.

For more details, visit [Intel® Distribution for Python*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html), [Intel® Extension for Scikit-Learn*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/scikit-learn.html) and [Purchase Prediction](https://github.com/oneapi-src/purchase-prediction).

## Solution Technical Details
In this section, we describe the data and how to replicate the results.

### ***Dataset***
The reference kit uses data from https://archive-beta.ics.uci.edu/ml/datasets/online+retail 

> *Please see this dataset's applicable license for terms and conditions. Intel® does not own the rights to this dataset and does not confer any rights to it.*

The dimension of the dataset is (541909, 8). Each record in the dataset represents purchase details of the product made by a customer with below eight features:
- InvoiceNo: Purchase Invoice Number
- StockCode: Product Stock Code
- Description: Product Description
- Quantity: Purchased Quantity
- InvoiceDate: Purchase Date
- UnitPrice: Unit price of the product
- CustomerID: Unique Customer ID
- Country: Country of the purchase

At first, the above features are used to create unique categories of products from the product stock code and product description corpus. Once the product categories are identified, every invoice purchase is re-arranged to include the category of the products bought. Based on the categories of the product the customer segmentation is created. Based on these categorical features (Product Category and Customer Segmentation) multiple classification models are developed that allow to predict the purchases that will be made by a new customer during their next visit, based on the segment on which the customer is expected to be. This will allow retailers to provide offers only for the predicted purchase.

### ***Hyperparameter Analysis***
In realistic scenarios, an analyst will run the same Machine Learning algorithm multiple times on the same dataset, scanning across different hyperparameters.  To capture this, we measure the total amount of time it takes to generate results across a grid of hyperparameters for a fixed algorithm, which we define as hyperparameter analysis.  In practice, the results of each hyperparameter analysis provides the analyst with many different customer segments and purchase predictions that they can take and further analyze.

The below table provides details about the hyperparameters & values used for hyperparameter tuning for each of the algorithms used in our benchmarking experiments:
| **Algorithm**                     | **Hyperparameters**
| :---                              | :---
| kNN                               | `space or parameters = pd.np.arange(1, 50, 1) ,`<br> `cross validation generator = 5 ,` <br> `n_jobs=-1` <br>
| Decision Tree Classifier          | `criterion = ['entropy', 'gini'] ,` <br> `'max_features' = ['sqrt', 'log2']` <br>
| Random Forest Classifier          | `criterion = ['gini'] ,` <br> `'n_estimators' =[20, 40, 60, 80, 100],` <br> `'max_features' = ['sqrt', 'log2']` <br>

## Validated Hardware Details
There are workflow-specific hardware and software setup requirements to run this use case.

| Recommended Hardware
| ----------------------------
| CPU: Intel® 2nd Gen Xeon® Platinum 8280 CPU @ 2.70GHz or higher
| RAM: 187 GB
| Recommended Free Disk Space: 20 GB or more

#### Minimal Requirements
* RAM: 64 GB total memory
* CPUs: 8
* Storage: 20GB
* Operating system: Ubuntu\* 22.04 LTS

## How it Works
The included code demonstrates a complete framework for:
1. Setting up a virtual environment for Intel®-accelerated ML.
2. Preprocessing data using Pandas and NLTK*.
3. Clustering data to create product categories using Intel® Extension for Scikit-learn*.
3. Training a kNN, Decision Tree or Random Forest Classifier model for customer segmentation using Intel® Extension for Scikit-learn*.
4. Predicting from the trained model on new data using Intel® Extension for Scikit-learn*.

## Get Started
Start by defining environment variables that will store the workspace, dataset and output paths. These directories will be used for all the commands executed using absolute paths.

[//]: # (capture: baremetal)
```bash
export WORKSPACE=$PWD/purchase-prediction
export DATA_DIR=$WORKSPACE/data
export OUTPUT_DIR=$WORKSPACE/output
```

### Download the Workflow Repository
Create a working directory for the workflow and clone the [Purchase Prediction](https://github.com/oneapi-src/purchase-prediction) repository into your working directory.

[//]: # (capture: baremetal)
```bash
mkdir -p $WORKSPACE && cd $WORKSPACE
```

```bash
git clone https://github.com/oneapi-src/purchase-prediction.git $WORKSPACE
```

[//]: # (capture: baremetal)
```bash
mkdir -p $DATA_DIR $OUTPUT_DIR
mkdir $OUTPUT_DIR/models $OUTPUT_DIR/logs
```

### Set Up Conda
To learn more, please visit [install anaconda on Linux](https://docs.anaconda.com/free/anaconda/install/linux/). 
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
rm Miniconda3-latest-Linux-x86_64.sh
```

### Set Up Environment
Install and set the libmamba solver as default solver. Do this by running the following commands:

```bash
conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba
```

The `$WORKSPACE/env/intel_env.yml` file contains all dependencies to create the intel environment necesary for runnig the workflow. 

| **Packages required in YAML file:**                 | **Version:**
| :---                          | :--
| `python`  | 3.9
| `intelpython3_full`  | 2024.0.0
| `pandas`  | 2.1.3
| `nltk`  | 3.8.1
| `xlsx2csv`  | 0.8.1

Execute next command to create the conda environment.

```bash
conda env create -f $WORKSPACE/env/intel_env.yml
```

Environment setup is required only once. This step does not cleanup the existing environment with the same name; make sure no conda environment exists with the same name. During this setup a new conda environment will be created with the dependencies listed in the YAML configuration.

Once the appropriate environment is created with the previous step then it has to be activated using the conda command as given below:
```bash
conda activate purchase_prediction_intel
```

### Download the Dataset
Execute the below commands to download the dataset and convert the excel file to a csv file.

[//]: # (capture: baremetal)
```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx --no-check-certificate -O $DATA_DIR/data.xlsx
xlsx2csv $DATA_DIR/data.xlsx $DATA_DIR/data.csv
```

> *Please see this dataset's applicable license for terms and conditions. Intel® does not own the rights to this dataset and does not confer any rights to it.*

## Supported Runtime Environment
You can execute the references pipelines using the following environments:
* Bare Metal

---
### Run Using Bare Metal
Follow these instructions to set up and run this workflow on your own development system.

#### Set Up System Software
Our examples use the `conda` package and environment on your local computer. If you don't already have `conda` installed or the `conda` environment created, go to [Set Up Conda*](#set-up-conda) or see the [Conda* Linux installation instructions](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html).

#### Run Workflow
As mentioned above, this hyperpersonalized target recommendation uses KNN, DecisionTreeClassifier and RandomForestClassifier from scikit learn library to train a AI model and generate labels for the passed in data. This process is captured within the `purchase-prediction-module.py` script. This script *reads and preprocesses the data*, and *performs hyperparameter analysis on either KNN, DecisionTreeClassifier or RandomForestClassifier*, while also reporting on the execution time for preprocessing and hyperparameter analysis steps.  Furthermore, this script can also save each of the intermediate models for an in-depth analysis of the quality of fit.  

This script mainly performs two objectives - Data preparation & Benchmarking of Intel® oneAPI Libraries.

Data preparation step is performed first. It involves the following:
1. Removing null values in the raw data
2. Analysing the description column by listing all the keywords & figuring out the most frequent occurring keywords
3. Defining the product categories
4. Using one hot encoding to create group of products
5. Creating customer categories

The script takes the following arguments:
```sh
usage: purchase-prediction-module.py [-h] [-rtd RAW_TRAIN_DATA] [-daf DATA_AUG_FACTOR] [-ftd FINAL_TRAIN_DATA] [-t TUNING] [-alg ALGORITHM]

optional arguments:
  -h, --help                                                    show this help message and exit
  -rtd RAW_TRAIN_DATA, --raw-train-data RAW_TRAIN_DATA          raw data csv file, if this parameter is specified then it will only perform the data preparation part
  -daf DATA_AUG_FACTOR, --data-aug-factor DATA_AUG_FACTOR       data augmentation/multiplication factor, requires --raw-train-data parameter
  -ftd FINAL_TRAIN_DATA, --final-train-data FINAL_TRAIN_DATA    final filtered data csv file, if this parameter is specified then it will skip the data preparation part
  -t TUNING, --tuning TUNING                                    hyperparameter tuning (0/1)
  -alg ALGORITHM, --algorithm ALGORITHM                         scikit learn classifier algorithm to be used (knn,dtc,rfc) - knn=KNearestNeighborClassifier, dtc=DecisionTreeClassifier, rfc=RandomForestClassifier
  -b BATCH_SIZE, --batch_size BATCH_SIZE                        batch size to run training without hyperparameter tuning
  -inf INFERENCE, --inference INFERENCE                         performs Inference on the saved models for batch data. Specify the model files i.e knn_model, dtc_model or rfc_model for knn=KNearestNeighborClassifier, dtc=DecisionTreeClassifier, rfc=RandomForestClassifier respectively
  -l LOGFILE, --logfile LOGFILE                                 log file to output benchmarking results to
```

**Data preparation**

Below command can be used to generate different dataset sizes which can be later used for feeding to the scikit-learn algorithms for benchmarking.

[//]: # (capture: baremetal)
```sh
python $WORKSPACE/src/purchase-prediction-module.py -rtd $DATA_DIR/data.csv -daf 20
```

The above example generates `$DATA_DIR/data_aug_20.csv` dataset which is the 20 fold multiplication of the initial filtered data.

**Training**

Use the below command to run the training with the generated data and default tuned hyperparameters for the algorithm KNeighborClassifier in intel environment.

[//]: # (capture: baremetal)
```sh
python $WORKSPACE/src/purchase-prediction-module.py -ftd $DATA_DIR/data_aug_20.csv -t 0 -alg knn
```

**Hyperparameter tuning**

In case of hyperparameter tuning mode training, use `-t 1` option. Gridsearch CV process is used for hyperparameter tuning.

[//]: # (capture: baremetal)
```sh
python $WORKSPACE/src/purchase-prediction-module.py -ftd $DATA_DIR/data_aug_20.csv -t 1 -alg knn
```

This will save the trained model in the path /model. i.e. knn_model.joblib will be saved in the path /model. Similarly, we would need to run the same for Decision Tree or Random Forest Classifier algorithms.

> Note: The tuned models will be saved as part of hyperparameter tuning and used for inference.

**Inference / Prediction**

Once the models are saved, we need to use them for inferencing. When the current source code is ran, batch inference is performed. The saved model is loaded & fed with a batchdata & the prediction output is obtained. The entire dataset file is passed to function for inference.

We need to pass the model name as a parameter in the console as shown below for KNearestNeighborClassifier model.

[//]: # (capture: baremetal)
```sh
python $WORKSPACE/src/purchase-prediction-module.py -ftd $DATA_DIR/data_aug_20.csv -inf knn_model -l $OUTPUT_DIR/logs/intel.log
```

You can use the following command to review the performance of the model:

[//]: # (capture: baremetal)
```sh
tail -n 4 $OUTPUT_DIR/logs/intel.log
```

**Note:** Inference benchmarking is done only for KNN model.

| **Algorithm**               | **Model Name to be passed**           
| :---                        | :---                            
| KNearestNeighborClassifier  | knn_model                         


#### Clean Up Bare Metal
Follow these steps to restore your ``$WORKSPACE`` directory to an initial step. Please note that all downloaded dataset files, conda environment, and logs created by the workflow will be deleted. Before executing next steps back up your important files.

```bash
conda deactivate
conda remove --name purchase_prediction_intel --all -y
```

[//]: # (capture: baremetal)
```bash
rm -r $DATA_DIR/*
rm -r $OUTPUT_DIR/*
```

## Expected Output
Below sample output would be generated by the training module which will capture the overall training time.
```
INFO:__main__:Running KNeighborsClassifier ...
INFO:__main__:====>KNeighborsClassifier Average Training Time with default hyperparameters 0.007310628890991211 secs
INFO:__main__:====> Program execution time 0.19111275672912598 secs
```

Below output would be generated by the training module which will capture the hyperparameter tuning training time.
```
INFO:__main__:====> KNeighborsClassifier Training Time with hyperparameter tuning 67.140428619385 secs
INFO:__main__:Saving the model ...
INFO:__main__:KNeighborsClassifier model 'knn_model.joblib' is saved in: /model
INFO:__main__:====> Program execution time 68.47586631774902 secs
```

Below output would be generated by the inference module which will capture the hyperparameter tuning inference time and accuracy.
```
kNN model loaded successfully
====> KNeighborsClassifier Model Inference Time is 0.20563340187072754 secs
====> Accuracy for kNN is: 100.0 % 
====> F1 score for kNN is: 1.0
====> Program execution time 1.6182811260223389 secs
```

## Summary and Next Steps
To build a customer segmentation solution at scale, Data Scientists will need to train models for substantial datasets and run inference more frequently.The ability to accelerate training will allow them to train more frequently and achieve better accuracy. Besides training, faster speed in inference will allow them to run prediction in real-time scenarios as well as more frequently. A Data Scientist will also look at data classification to tag and categorize data so that it can be better understood and analyzed. This task requires a lot of training and retraining, making the job tedious. The ability to get it faster speed will accelerate the ML pipeline. This reference kit implementation provides performance-optimized guide around customer purchase prediction use cases that can be easily scaled across similar use cases.

## Learn More
For more information about or to read about other relevant workflow examples, see these guides and software resources:

- [Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)
- [Intel® Distribution for Python*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html)
- [Intel® Extension for Scikit-Learn*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/scikit-learn.html)

## Support
If you have questions or issues about this use case, want help with troubleshooting, want to report a bug or submit enhancement requests, please submit a GitHub issue.

## Appendix
\*Names and brands that may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html).

### Reference
<a id="fabien_2018">[1]</a> Fabien, D. (2018). "Customer Segmentation". Kaggle. Found in: https://www.kaggle.com/code/fabiendaniel/customer-segmentation/

### Disclaimers
To the extent that any public or non-Intel® datasets or models are referenced by or accessed using tools or code on this site those datasets or models are provided by the third party indicated as the content source. Intel® does not create the content and does not warrant its accuracy or quality. By accessing the public content, or using materials trained on or with such content, you agree to the terms associated with that content and that your use complies with the applicable license.

Intel® expressly disclaims the accuracy, adequacy, or completeness of any such public content, and is not liable for any errors, omissions, or defects in the content, or for any reliance on the content. Intel® is not liable for any liability or damages relating to your use of public content.
