import getpass
import json

from src.Restrictions import Restriction, RestrictionKind
from src.notebook_code.data_preprocessing import preprocess_data
from src.notebook_code.model_train_eval import train_and_eval_models

if __name__ == "__main__":
    # CONFIG
    NUM_TFIDF_FEATURES = 300
    TFIDF_NAMES = [f"diagnoses_tfidf_{pos}" for pos in range(NUM_TFIDF_FEATURES)]
    RUN_PREPROCESSING = False
    TRAIN_AND_EVAL_MODELS = True
    CF_RESTRICTIONS = [
                          Restriction(RestrictionKind.DRUG_DRUG, "225975", "3422")
                      ] + [
                          Restriction(RestrictionKind.DRUG_DISEASE, "225975", "V1582"),
                          Restriction(RestrictionKind.DRUG_DISEASE, "4514", "V1582")
                      ]

    DATA_OUT_FOLDER = "./processed_mimic_data"
    DRG_MODEL_PATH = "./drg_models/all_updated"
    MODELS_TO_TRAIN = [
        # 'dynamic_lstm',
        #'full_lstm',
        'rf',
        '1-NN'
    ]
    MODELS_TO_EXPLAIN = 'rf'
    RESULTS_PATH = "./results"

    personal_config = json.load(open("./personal_config.json"))
    postgres_pw = personal_config['postgres_pw']

    # NOTEBOOK 1
    if RUN_PREPROCESSING:
        print("Started Data Preprocessing step...")

        preprocess_data(
            postgres_pw=postgres_pw,
            data_path=DATA_OUT_FOLDER
        )
        print("Data preprocessing done!")

    # NOTEBOOK 2
    if TRAIN_AND_EVAL_MODELS:
        print("Started model training and evaluation step...")
        train_and_eval_models(
            postgres_pw=postgres_pw,
            data_path=DATA_OUT_FOLDER,
            drg_model_path=DRG_MODEL_PATH,
            models_to_train=MODELS_TO_TRAIN,
            model_to_explain=MODELS_TO_EXPLAIN,
            results_path=RESULTS_PATH,
            sequences_to_plot=[1, 2, 3, 4, 5, 6, 7, 8],
            tfidf_names=TFIDF_NAMES,
            full_batch_size=2048,
            dynamic_batch_size=2048,
            cf_restrictions=CF_RESTRICTIONS
        )
        print("Model training and evaluation step done!")
