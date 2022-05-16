# Comparison of approaches for predicting ICU survival of patients in the MIMIC-III clinical database and evaluation of different methods for producing counterfactual samples.

This repository is forked from, and based on [this repository](https://github.com/zhendong3wang/counterfactuals-for-event-sequences)

Please refer to the original publication:
```
@inproceedings{wang_counterfactual_2021,
	title = {Counterfactual {Explanations} for {Survival} {Prediction} of {Cardiovascular} {ICU} {Patients}},
	doi = {10.1007/978-3-030-77211-6_38},
	booktitle = {Artificial {Intelligence} in {Medicine}},
	author = {Wang, Zhendong and Samsten, Isak and Papapetrou, Panagiotis},
	year = {2021},
	pages = {338--348},
}
```

## MIMIC-III data 
[MIMIC-III](https://mimic.physionet.org/gettingstarted/overview/) is a public data set comprising data of ICU patients.
To be able to access it, a free accreditation process is required.
Therefore, this data is not included in this repository.

The MIMIC-III documentation provides instruction for how to download and pre-process the data.
For the code in this repository to run the MIMIC-III data has to be stored in a POSTGRES database
as described in the MIMIC-III documentation.

## Running the experiments performed in the thesis

### Installation

Make sure to have Python 3.7 installed and run:

`pip install -r requirements.txt`

### Configuration

The `./personal_config.json` file has to contain the password to the postgres database which stores the 
MIMIC-III data.

### Train DRG models

Run the following two commands to train the DRG models described in the thesis.

```
python train.py --config configs/medseq_config_del.json
python train.py --config configs/medseq_config_ret.json
```

### Running the experiments

Run the following command to perform the experiments described in the thesis:

```
python  run_experiments.py
```

See the file contents for configuration options.


## Contact

If any issues or questions come up do not hesitate to contact the author at <flo[dot]bauernfeind[at]gmail[dot]com>!

