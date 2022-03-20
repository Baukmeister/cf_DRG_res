from datetime import datetime
import os
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import keras
import tensorflow as tf
import random as python_random
from keras import layers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import LSTM, CuDNNLSTM, Dense, Dropout, Input, ConvLSTM2D, Flatten, Add, Concatenate, Dot, \
    Multiply
from keras.layers import Maximum, Average, Activation
from tensorflow.keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils.vis_utils import plot_model
import math
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from keras import backend as K
import pandas.io.sql as sqlio
import psycopg2
import getpass
import matplotlib.pyplot as plt
from src.visualize import plot_sequence


def train_and_eval_models(
        postgres_pw,
        data_path,
        drg_model_path,
        models_to_train,
        model_to_explain,
        results_path
):
    tf.test.is_gpu_available()

    train_pos = pd.read_csv(f"{data_path}/train_pos.txt")
    train_neg = pd.read_csv(f"{data_path}/train_neg.txt")

    train_pos['survival'] = [1 for i in range(train_pos.shape[0])]
    train_neg['survival'] = [0 for i in range(train_neg.shape[0])]

    static_feature_names = [
        'gender',
        'age',
        'los_hospital',
        'ethnicity',
        'admission_type',
        'first_hosp_stay',
        'first_icu_stay'
    ]

    result_df = pd.DataFrame(columns=["metric", "value"])


    train = pd.concat([train_pos, train_neg]).reset_index()[['survival', 'events'] + static_feature_names].dropna()
    train_reordered = train.sample(frac=1, random_state=3)

    train_reordered[static_feature_names] = preprocessing.MinMaxScaler().fit_transform(
        train_reordered[static_feature_names].values)

    X_train_events, X_train_static, y_train = train_reordered['events'], train_reordered[static_feature_names], \
                                              train_reordered['survival']

    validation_pos = pd.read_csv(f"{data_path}/validation_pos.txt")
    validation_neg = pd.read_csv(f"{data_path}/validation_neg.txt")

    validation_pos['survival'] = [1 for i in range(validation_pos.shape[0])]
    validation_neg['survival'] = [0 for i in range(validation_neg.shape[0])]

    #    validation_neg['events'] = validation_neg['drug_events'] + " " + validation_neg['procedure_codes']
    #    validation_pos['events'] = validation_pos['drug_events'] + " " + validation_pos['procedure_codes']
    #
    #    train_neg['events'] = train_neg['drug_events'] + " " + train_neg['procedure_codes']
    #    train_pos['events'] = train_pos['drug_events'] + " " + train_pos['procedure_codes']

    validation = pd.concat([validation_pos, validation_neg]).reset_index()
    validation_reordered = validation.sample(frac=1, random_state=3)

    X_val_neg_static = validation_neg[static_feature_names]
    X_val_events, X_val_static, y_val = validation_reordered["events"], validation_reordered[static_feature_names], \
                                        validation_reordered['survival']

    X_val_events.head()

    vocab_size = 1100
    max_seq_length = 74

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(X_train_events)

    X_train_sequences = tokenizer.texts_to_sequences(X_train_events)
    X_val_sequences = tokenizer.texts_to_sequences(X_val_events)

    print(f'Before texts_to_sequences():\n {X_train_events.iloc[0]}\n')

    print(f'After texts_to_sequences():\n {X_train_sequences[0]}')

    X_train_padded = sequence.pad_sequences(X_train_sequences, maxlen=max_seq_length, padding='post')
    X_val_padded = sequence.pad_sequences(X_val_sequences, maxlen=max_seq_length, padding='post')

    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_' + string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_' + string])
        plt.show()

    def eval_model_preds(preds, reference):

        validation_acc = accuracy_score(y_true=reference, y_pred=preds)
        f1 = f1_score(y_true=reference, y_pred=preds)
        print(f'Validation Accuracy: {validation_acc}')
        print()

        print(f'F1 Score: {f1}')
        print()

        confusion_matrix_df = pd.DataFrame(
            confusion_matrix(y_true=reference, y_pred=preds, labels=[1, 0]),
            index=['True:pos', 'True:neg'],
            columns=['Pred:pos', 'Pred:neg']
        )
        print('Confusion Matrix:')
        print(confusion_matrix_df)
        print()

        print('Negative and positive predictions')
        print(pd.value_counts(preds))

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    seed_value = 3

    os.environ['PYTHONHASHSEED'] = str(seed_value)

    np.random.seed(seed_value)

    python_random.seed(seed_value)

    tf.random.set_seed(seed_value)

    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

    def reset_seeds(seed_value=3):
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        np.random.seed(seed_value)
        python_random.seed(seed_value)
        tf.random.set_seed(seed_value)

    reset_seeds()

    inputs = keras.Input(shape=(None,), dtype="int32")

    x = layers.Embedding(vocab_size, 128)(inputs)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dense(64)(x)
    x = layers.Dense(64)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)
    dynamic_lstm_model = keras.Model(inputs, outputs)

    plot_model(dynamic_lstm_model)

    dynamic_lstm_model.summary()

    if 'dynamic_lstm' in models_to_train:
        dynamic_lstm_model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
        reset_seeds()

        model_history = dynamic_lstm_model.fit(
            X_train_padded,
            y_train,
            epochs=30,
            batch_size=64,
            validation_data=(X_val_padded, y_val),
            callbacks=[early_stopping]
        )

        plot_graphs(model_history, "accuracy")
        plot_graphs(model_history, "loss")

        dynamic_lstm_pred = np.array([1 if pred > 0.5 else 0 for pred in dynamic_lstm_model.predict(X_val_padded)])
        eval_model_preds(dynamic_lstm_pred, y_val)

    seed_value = 3

    os.environ['PYTHONHASHSEED'] = str(seed_value)

    np.random.seed(seed_value)

    python_random.seed(seed_value)

    tf.random.set_seed(seed_value)

    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

    reset_seeds()

    x_i_dyn = Input(shape=(None,), dtype="int32")
    x = layers.Embedding(vocab_size, 128)(x_i_dyn)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x_pa = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dense(64)(x_pa)
    x_dyn = layers.Dense(64)(x)

    x_i_stat = Input(shape=(len(static_feature_names),))
    x2 = layers.Dense(64)(x_i_stat)
    x2 = layers.Activation('relu')(x2)
    x_pa2 = layers.Dense(64)(x2)
    x2 = Activation('relu')(x_pa2)
    x_stat = Dropout(0.1)(x2)

    x_dot = Dot(axes=1, normalize=True)([x_stat, x_dyn])
    x = layers.Dense(128)(x_dot)
    x = Activation('relu')(x)
    x = layers.Dense(128)(x)
    x = layers.Dense(128)(x)
    x = Activation('relu')(x)
    x = layers.Dense(128)(x)

    x_o = Dense(1, activation="sigmoid")(x)

    full_lstm_model = Model([x_i_dyn, x_i_stat], x_o)

    full_lstm_model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

    plot_model(full_lstm_model)

    full_lstm_model.summary()

    if 'full_lstm' in models_to_train:
        full_lstm_hist = full_lstm_model.fit(
            [X_train_padded, X_train_static],
            y_train,
            epochs=100,
            batch_size=64,
            validation_data=([X_val_padded, X_val_static], y_val),
            callbacks=early_stopping
        )

        plot_graphs(full_lstm_hist, "accuracy")
        plot_graphs(full_lstm_hist, "loss")

        full_lstm_pred = np.array(
            [1 if pred > 0.5 else 0 for pred in full_lstm_model.predict([X_val_padded, X_val_static])])
        eval_model_preds(full_lstm_pred, y_val)

    if 'rf' in models_to_train:
        rf = RandomForestClassifier()
        rf.fit(X_train_static, y_train)
        rf_pred = rf.predict(X_val_static)
        eval_model_preds(rf_pred, y_val)

    if model_to_explain == 'dynamic_lstm':
        y_pred = dynamic_lstm_pred
        model = dynamic_lstm_model
    if model_to_explain == 'full_lstm':
        y_pred = full_lstm_pred
        model = full_lstm_model
    if model_to_explain == 'rf':
        y_pred = rf_pred
        model = rf

    X_pred_negative = X_val_padded[y_pred == 0]
    X_pred_negative_static = X_val_static[y_pred == 0]

    original_event_sequences = tokenizer.sequences_to_texts(X_pred_negative)

    pd.DataFrame(original_event_sequences).to_csv(path_or_buf=f'{data_path}/test_neg.txt', index=False, header=False,
                                                  sep=' ', quoting=csv.QUOTE_NONE, escapechar=' ')

    trans_results_delete = pd.read_csv(f'{drg_model_path}/drg_delete/preds', header=None)

    X_cf_delete = tokenizer.texts_to_sequences(trans_results_delete[0])

    X_cf_delete_padded = sequence.pad_sequences(X_cf_delete, maxlen=max_seq_length, padding='post')

    delete_generate_results = pd.read_csv(f'{drg_model_path}/drg_delete_retrieve/preds', header=None)

    X_cf_delete_retrieve = tokenizer.texts_to_sequences(delete_generate_results[0])

    X_cf_delete_retrieve_padded = sequence.pad_sequences(X_cf_delete_retrieve, maxlen=max_seq_length, padding='post')

    nn_model = NearestNeighbors(n_neighbors=1, metric='hamming')
    target_label = 1
    X_target_label = X_train_padded[y_train == target_label]

    nn_model.fit(X_target_label)

    closest = nn_model.kneighbors(X_pred_negative, return_distance=False)
    trans_results_nn = X_target_label[closest[:, 0]]

    X_cf_one_nn = trans_results_nn

    trans_event_delete = tokenizer.sequences_to_texts(X_cf_delete_padded)
    trans_event_delete_retrieve = tokenizer.sequences_to_texts(X_cf_delete_retrieve_padded)
    trans_event_one_nn = tokenizer.sequences_to_texts(X_cf_one_nn)

    test_size = X_val_neg_static.shape[0]

    if model_to_explain == 'full_lstm':
        fraction_success = np.sum(model.predict([X_cf_delete_padded, X_val_neg_static]) > 0.5) / test_size
    elif model_to_explain == "rf":
        fraction_success = np.sum(model.predict(X_val_neg_static) > 0.5) / test_size
    else:
        fraction_success = np.sum(model.predict(X_cf_delete_padded) > 0.5) / test_size
    print(round(fraction_success, 4))

    result_df.append(
        {
            'metric': 'frac_delete',
            'value': fraction_success
        },
        ignore_index=True
    )

    if model_to_explain == 'full_lstm':
        fraction_success = np.sum(
            model.predict([X_cf_delete_retrieve_padded[:len(X_val_neg_static)], X_val_neg_static]) > 0.5) / test_size
    elif model_to_explain == "rf":
        fraction_success = np.sum(model.predict(X_val_neg_static) > 0.5) / test_size
    else:
        fraction_success = np.sum(model.predict(X_cf_delete_retrieve_padded) > 0.5) / test_size
    print(round(fraction_success, 4))

    result_df = result_df.append(
        {
            'metric': 'frac_delete_retrieve',
            'value': fraction_success,
        },
        ignore_index=True,
    )

    if model_to_explain == 'full_lstm':
        fraction_success = np.sum(model.predict([X_cf_one_nn, X_pred_negative_static]) > 0.5) / test_size
    elif model_to_explain == "rf":
        fraction_success = np.sum(model.predict(X_val_neg_static) > 0.5) / test_size
    else:
        fraction_success = np.sum(model.predict(X_cf_one_nn) > 0.5) / test_size
    print(round(fraction_success, 4))

    result_df.append(
        {
            'metric': 'frac_one_nn',
            'value': fraction_success
        },
        ignore_index=True
    )

    clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
    clf.fit(X_train_padded)

    y_pred_val = clf.predict(X_val_padded)

    n_error_val = y_pred_val[y_pred_val == -1].size

    validation_size = X_val_padded.shape[0]
    outlier_score_val = n_error_val / validation_size

    y_pred_test = clf.predict(X_cf_delete_padded)
    n_error_test = y_pred_test[y_pred_test == -1].size

    outlier_score_delete = n_error_test / test_size
    print(round(outlier_score_delete, 4))

    result_df = result_df.append(
        {
            'metric': 'lof_delete',
            'value': outlier_score_delete
        },
        ignore_index=True
    )
    y_pred_test2 = clf.predict(X_cf_delete_retrieve_padded)
    n_error_test2 = y_pred_test2[y_pred_test2 == -1].size

    outlier_score_delete_retrieve = n_error_test2 / test_size
    print(round(outlier_score_delete_retrieve, 4))

    result_df = result_df.append(
        {
            'metric': 'lof_delete_retrieve',
            'value': outlier_score_delete_retrieve
        },
        ignore_index=True
    )

    y_pred_test3 = clf.predict(X_cf_one_nn)
    n_error_test3 = y_pred_test3[y_pred_test3 == -1].size

    outlier_score_one_nn = n_error_test3 / test_size
    print(round(outlier_score_one_nn, 4))

    result_df = result_df.append(
        {
            'metric': 'lof_one_nn',
            'value': outlier_score_one_nn
        },
        ignore_index=True
    )



    chencherry = SmoothingFunction()

    def get_pairwise_bleu(original, transformed):

        results = [sentence_bleu(
            references=[pair[0].split()],
            hypothesis=pair[1].split(),
            weights=[0.25, 0.25, 0.25, 0.25],
            smoothing_function=chencherry.method1)
            for pair in zip(original, transformed)]

        return results

    pairwise_bleu_delete = get_pairwise_bleu(original_event_sequences, trans_event_delete)
    avg_bleu_delete = sum(pairwise_bleu_delete) / test_size
    print(round(avg_bleu_delete, 4))

    result_df = result_df.append(
        {
            'metric': 'avg_bleu_delete',
            'value': avg_bleu_delete
        },
        ignore_index=True
    )

    pairwise_bleu_delete_retrieve = get_pairwise_bleu(original_event_sequences, trans_event_delete_retrieve)
    avg_bleu_delete_retrieve = sum(pairwise_bleu_delete_retrieve) / test_size
    print(round(avg_bleu_delete_retrieve, 4))

    result_df = result_df.append(
        {
            'metric': 'avg_bleu_delete_retrieve',
            'value': avg_bleu_delete_retrieve
        },
        ignore_index=True
    )

    pariwise_bleu_one_nn = get_pairwise_bleu(original_event_sequences, trans_event_one_nn)
    avg_bleu_one_nn = sum(pariwise_bleu_one_nn) / test_size
    print(round(avg_bleu_one_nn, 4))

    result_df = result_df.append(
        {
            'metric': 'avg_bleu_one_nn',
            'value': avg_bleu_one_nn
        },
        ignore_index=True
    )


    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))

    plt.sca(ax[0])
    plt.title('DeleteOnly, BLUE score')
    plt.hist(pairwise_bleu_delete, density=True, bins=30)

    plt.sca(ax[1])
    plt.title('DeleteAndRetrieve, BLUE score')
    plt.hist(pairwise_bleu_delete_retrieve, density=True, bins=30)

    plt.sca(ax[2])
    plt.title('1-NN, BLUE score')
    plt.hist(pariwise_bleu_one_nn, density=True, bins=30)

    plt.show()

    original_counts = pd.DataFrame(columns=['total', 'drug', 'procedure'])

    def get_counts_table(event_sequences):
        temp_list = list()
        for seq in event_sequences:
            splitted = seq.split()
            total = len(splitted)

            drug = len([x for x in splitted if int(x) >= 220000])
            procedure = total - drug

            temp_list.append({'total': total, 'drug': drug, 'procedure': procedure})

        return pd.DataFrame(temp_list)

    df_original_counts = get_counts_table(original_event_sequences)

    df_original_counts.head()

    trans_counts_delete = get_counts_table(trans_event_delete)
    trans_counts_delete_retrieve = get_counts_table(trans_event_delete_retrieve)
    trans_counts_one_nn = get_counts_table(trans_event_one_nn)

    substracted_delete = trans_counts_delete.subtract(df_original_counts)
    substracted_delete_retrieve = trans_counts_delete_retrieve.subtract(df_original_counts)
    substracted_one_nn = trans_counts_one_nn.subtract(df_original_counts)

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(16, 12))

    plt.sca(ax[0, 0])
    plt.title('DeleteOnly, total difference')
    plt.hist(substracted_delete['total'], density=True, bins=30)

    plt.sca(ax[0, 1])
    plt.title('DeleteOnly, drug event difference')
    plt.hist(substracted_delete['drug'], density=True, bins=30)

    plt.sca(ax[0, 2])
    plt.title('DeleteOnly, procedure difference')
    plt.hist(substracted_delete['procedure'], density=True, bins=12)

    plt.sca(ax[1, 0])
    plt.title('DeleteAndRetrieve, total difference')
    plt.hist(substracted_delete_retrieve['total'], density=True, bins=30)

    plt.sca(ax[1, 1])
    plt.title('DeleteAndRetrieve, drug event difference')
    plt.hist(substracted_delete_retrieve['drug'], density=True, bins=30)

    plt.sca(ax[1, 2])
    plt.title('DeleteAndRetrieve, procedure difference')
    plt.hist(substracted_delete_retrieve['procedure'], density=True, bins=12)

    plt.sca(ax[2, 0])
    plt.title('1-NN, total difference')
    plt.hist(substracted_one_nn['total'], density=True, bins=30)

    plt.sca(ax[2, 1])
    plt.title('1-NN, drug event difference')
    plt.hist(substracted_one_nn['drug'], density=True, bins=30)

    plt.sca(ax[2, 2])
    plt.title('1-NN, procedure difference')
    plt.hist(substracted_one_nn['procedure'], density=True, bins=12)

    plt.show()

    conn = psycopg2.connect(
        database="mimic",
        user='postgres',
        password=postgres_pw,
        host="127.0.0.1",
        port="5432",
        options=f'-c search_path=mimiciii')

    itemid_to_name = pd.read_sql(
        """
        SELECT itemid, abbreviation, label
        FROM d_items;
        """, conn)

    itemid_to_name = itemid_to_name[itemid_to_name['itemid'] >= 220000]

    itemid_to_name2 = pd.read_sql(
        """
        SELECT icd9_code, short_title, long_title
        FROM d_icd_procedures;
        """, conn)

    itemid_to_name2.head()

    itemid_to_name2 = itemid_to_name2.rename(
        columns={'icd9_code': 'itemid', 'short_title': 'abbreviation', 'long_title': 'label'}
    )

    itemid_to_name_concat = pd.concat([itemid_to_name, itemid_to_name2])

    itemid_to_name_concat['label'] = itemid_to_name_concat['label'].astype('str')
    itemid_to_name_concat['itemid'] = itemid_to_name_concat['itemid'].astype('int')

    def code_to_name(event_sequence):
        code_sequence = [int(event) for event in event_sequence.split()]

        temp_list = list()
        for code in code_sequence:
            event_name = itemid_to_name_concat[itemid_to_name_concat['itemid'] == code]['label'].item()
            temp_list.append(event_name)

        return temp_list

    ## EXPORT
    output_folder = f"{results_path}/{datetime.now().strftime('%Y_%m_%d-%H-%M-%S')}-{model_to_explain}"
    os.mkdir(output_folder)
    result_df.to_csv(f"{output_folder}/CF_metrics.csv")

    sample_id = 2

    code_to_name(original_event_sequences[sample_id])

    plot_sequence(
        code_to_name(original_event_sequences[sample_id]),
        title=f"{model_to_explain} - Original Sequence",
        output_folder=output_folder
    )

    code_to_name(trans_event_delete[sample_id])

    plot_sequence(
        code_to_name(trans_event_delete[sample_id]),
        title=f"{model_to_explain} - DRG: DELETE Sequence",
        output_folder=output_folder
    )

    code_to_name(trans_event_delete_retrieve[sample_id])

    plot_sequence(
        code_to_name(trans_event_delete_retrieve[sample_id]),
        title=f"{model_to_explain} - DRG: DELETE-RETRIEVE Sequence",
        output_folder=output_folder
    )

    code_to_name(trans_event_one_nn[sample_id])

    plot_sequence(
        code_to_name(trans_event_one_nn[sample_id]),
        title=f"{model_to_explain} - 1-NN Sequence",
        output_folder=output_folder
    )
