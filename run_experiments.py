import getpass
import json

from src.Restrictions import Restriction, RestrictionKind
from src.notebook_code.data_preprocessing import preprocess_data
from src.notebook_code.model_train_eval import train_and_eval_models

if __name__ == "__main__":
    # CONFIG
    NUM_TFIDF_FEATURES = 300
    RNG_SEED = 102
    TFIDF_NAMES = [f"diagnoses_tfidf_{pos}" for pos in range(NUM_TFIDF_FEATURES)]
    RUN_PREPROCESSING = False
    TRAIN_AND_EVAL_MODELS = True
    CF_RESTRICTIONS = [
                          # Fentanyl + Morphine
                          Restriction(RestrictionKind.DRUG_DRUG, "221744", "225154"),
                          Restriction(RestrictionKind.DRUG_DRUG, "225942", "225154"),

                          # Furosemide + Vancomycin
                          Restriction(RestrictionKind.DRUG_DRUG, "225942", "225798"),

                          # Furosemide + Fentanyl
                          Restriction(RestrictionKind.DRUG_DRUG, "225942", "221744"),
                          Restriction(RestrictionKind.DRUG_DRUG, "225942", "225942"),

                          # Furosemide + Morphine
                          Restriction(RestrictionKind.DRUG_DRUG, "225942", "225154"),

                          # Fentanyl + Propofol
                          Restriction(RestrictionKind.DRUG_DRUG, "221744", "222168"),
                          Restriction(RestrictionKind.DRUG_DRUG, "225942", "222168"),
                          # Morphine + Propofol
                          Restriction(RestrictionKind.DRUG_DRUG, "225154", "222168"),

                          # Norephinephrine + Insulin
                          Restriction(RestrictionKind.DRUG_DRUG, "221906", "223257"),
                          Restriction(RestrictionKind.DRUG_DRUG, "221906", "223260"),
                          Restriction(RestrictionKind.DRUG_DRUG, "221906", "223262"),
                          Restriction(RestrictionKind.DRUG_DRUG, "221906", "223261"),
                          Restriction(RestrictionKind.DRUG_DRUG, "221906", "223259"),
                          Restriction(RestrictionKind.DRUG_DRUG, "221906", "223258"),

                          # Phenylephrine + Insulin
                          Restriction(RestrictionKind.DRUG_DRUG, "221749", "223257"),
                          Restriction(RestrictionKind.DRUG_DRUG, "221749", "223260"),
                          Restriction(RestrictionKind.DRUG_DRUG, "221749", "223262"),
                          Restriction(RestrictionKind.DRUG_DRUG, "221749", "223261"),
                          Restriction(RestrictionKind.DRUG_DRUG, "221749", "223259"),
                          Restriction(RestrictionKind.DRUG_DRUG, "221749", "223258")

                      ] + [
                          # Metoprolol (Beta-Blocker) + Heart Block
                          Restriction(RestrictionKind.DRUG_DISEASE, "225974", "4260"),
                          Restriction(RestrictionKind.DRUG_DISEASE, "225974", "4263"),
                          Restriction(RestrictionKind.DRUG_DISEASE, "225974", "4264"),
                          Restriction(RestrictionKind.DRUG_DISEASE, "225974", "42611"),
                          Restriction(RestrictionKind.DRUG_DISEASE, "225974", "42613"),
                          Restriction(RestrictionKind.DRUG_DISEASE, "225974", "42612"),
                          Restriction(RestrictionKind.DRUG_DISEASE, "225974", "42652"),
                          Restriction(RestrictionKind.DRUG_DISEASE, "225974", "42610"),
                          Restriction(RestrictionKind.DRUG_DISEASE, "225974", "42653"),
                          Restriction(RestrictionKind.DRUG_DISEASE, "225974", "4266"),
                          Restriction(RestrictionKind.DRUG_DISEASE, "225974", "4262"),
                          Restriction(RestrictionKind.DRUG_DISEASE, "225974", "74686"),
                          Restriction(RestrictionKind.DRUG_DISEASE, "225974", "42651"),
                          Restriction(RestrictionKind.DRUG_DISEASE, "225974", "42650")
                      ]

    DATA_OUT_FOLDER = "./processed_mimic_data"
    DRG_MODEL_PATH = "./drg_models/all_updated"
    MODELS_TO_TRAIN = [
        'dynamic_lstm',
        'full_lstm',
        'rf',
        '1-NN'
    ]
    MODELS_TO_EXPLAIN = 'full_lstm'
    RESULTS_PATH = "./results"

    personal_config = json.load(open("./personal_config.json"))
    postgres_pw = personal_config['postgres_pw']

    # NOTEBOOK 1
    if RUN_PREPROCESSING:
        print("Started Data Preprocessing step...")

        preprocess_data(
            postgres_pw=postgres_pw,
            data_path=DATA_OUT_FOLDER,
            num_tfidf_features=NUM_TFIDF_FEATURES,
            tfidf_names=TFIDF_NAMES,
            seed=RNG_SEED
        )
        print("Data preprocessing done!")

    # NOTEBOOK 2
    if TRAIN_AND_EVAL_MODELS:
        print("Started model training and evaluation step...")
        for batch_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            train_and_eval_models(
                postgres_pw=postgres_pw,
                data_path=DATA_OUT_FOLDER,
                drg_model_path=DRG_MODEL_PATH,
                models_to_train=MODELS_TO_TRAIN,
                model_to_explain=MODELS_TO_EXPLAIN,
                results_path=RESULTS_PATH,
                sequences_to_plot=[0, 2, 6, 14, 37, 40, 69, 72],
                tfidf_names=TFIDF_NAMES,
                full_batch_size=batch_size,
                dynamic_batch_size=batch_size,
                epochs=40,
                cf_restrictions=CF_RESTRICTIONS,
                early_stopping=True,
                seed=RNG_SEED
            )
        print("Model training and evaluation step done!")
