from datetime import datetime
import os
import csv
from typing import List

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor, KNeighborsClassifier
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
from keras import backend as K
import pandas.io.sql as sqlio
import psycopg2
import getpass
import matplotlib.pyplot as plt

from src.Restrictions import Restriction
from src.visualize import plot_sequence


def train_and_eval_models(
        postgres_pw,
        data_path,
        drg_model_path,
        models_to_train,
        model_to_explain,
        results_path,
        sequences_to_plot,
        tfidf_names,
        dynamic_batch_size=64,
        full_batch_size=8,
        epochs=50,
        cf_restrictions=List[Restriction],
        early_stopping=True,
        seed=42
):
    tf.test.is_gpu_available()
    restriction_modes = ["cf_restrictions", "no_cf_restrictions"]

    train_pos = pd.read_csv(f"{data_path}/train_pos.txt")
    train_neg = pd.read_csv(f"{data_path}/train_neg.txt")

    train_pos['survival'] = [1 for i in range(train_pos.shape[0])]
    train_neg['survival'] = [0 for i in range(train_neg.shape[0])]



    output_folder = f"{results_path}/{datetime.now().strftime('%Y_%m_%d-%H-%M-%S')}-{model_to_explain}-dbs-{dynamic_batch_size}-fbs-{full_batch_size}-seed-{seed}"

    if not early_stopping:
        output_folder += "-no_early_stopping"

    os.mkdir(output_folder)

    plot_folder = f"{output_folder}/plots"
    os.mkdir(plot_folder)

    static_feature_names = [
                               'gender',
                               'age',
                               'los_hospital',
                               'ethnicity',
                               'admission_type',
                               'first_hosp_stay',
                               'first_icu_stay'
                           ] + tfidf_names

    prediction_result_df = pd.DataFrame(columns=["metric", "value"])

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

    validation = pd.concat([validation_pos, validation_neg]).reset_index()
    validation_reordered = validation.sample(frac=1, random_state=3)

    X_val_neg_static = validation_neg[static_feature_names]
    X_val_events, X_val_static, y_val = validation["events"], validation[static_feature_names], \
                                        validation['survival']

    X_val_events.head()

    vocab_size = 3000
    max_seq_length = 74

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(X_train_events)

    X_train_sequences = tokenizer.texts_to_sequences(X_train_events)
    X_val_sequences = tokenizer.texts_to_sequences(X_val_events)

    print(f'Before texts_to_sequences():\n {X_train_events.iloc[0]}\n')

    print(f'After texts_to_sequences():\n {X_train_sequences[0]}')

    X_train_padded = sequence.pad_sequences(X_train_sequences, maxlen=max_seq_length, padding='post')
    X_val_padded = sequence.pad_sequences(X_val_sequences, maxlen=max_seq_length, padding='post')

    def plot_graphs(history, metric, title):
        plt.clf()
        plt.plot(history.history[metric])
        plt.plot(history.history['val_' + metric])
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend([metric, 'val_' + metric])
        plt.title(f"Training performance for {title}")
        plt.savefig(f"{plot_folder}/training_performance-{title}_{metric}.png")

    def eval_model_preds(preds, reference, prediction_result_df, model_name):

        categoric_preds = np.array([1 if pred > 0.5 else 0 for pred in preds])

        validation_acc = accuracy_score(y_true=reference, y_pred=categoric_preds)
        f1 = f1_score(y_true=reference, y_pred=categoric_preds)
        precision = precision_score(y_true=reference, y_pred=categoric_preds)
        recall = recall_score(y_true=reference, y_pred=categoric_preds)
        fpr, tpr, thresholds = roc_curve(reference, preds, pos_label=1)
        auc_score = auc(fpr, tpr)

        print(f'Validation Accuracy: {validation_acc}')

        prediction_result_df = prediction_result_df.append(
            {
                'metric': f'{model_name}_val_accuracy',
                'value': validation_acc
            },
            ignore_index=True
        )

        print(f'F1 Score: {f1}')
        prediction_result_df = prediction_result_df.append(
            {
                'metric': f'{model_name}_f1',
                'value': f1
            },
            ignore_index=True
        )

        print(f'Precision: {precision}')
        prediction_result_df = prediction_result_df.append(
            {
                'metric': f'{model_name}_precision',
                'value': precision
            },
            ignore_index=True
        )

        print(f'Recall: {recall}')
        prediction_result_df = prediction_result_df.append(
            {
                'metric': f'{model_name}_recall',
                'value': recall
            },
            ignore_index=True
        )

        print(f'AUC: {auc_score}')
        prediction_result_df = prediction_result_df.append(
            {
                'metric': f'{model_name}_auc',
                'value': auc_score
            },
            ignore_index=True
        )

        confusion_matrix_df = pd.DataFrame(
            confusion_matrix(y_true=reference, y_pred=categoric_preds, labels=[1, 0]),
            index=['True:pos', 'True:neg'],
            columns=['Pred:pos', 'Pred:neg']
        )
        print('Confusion Matrix:')
        print(confusion_matrix_df)
        print()

        print('Negative and positive predictions')
        print(pd.value_counts(categoric_preds))

        return prediction_result_df

    if early_stopping:
        callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
    else:
        callbacks = []

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

    plot_model(dynamic_lstm_model, f"{plot_folder}/dynamic_lstm_model.png")

    dynamic_lstm_model.summary()

    if 'dynamic_lstm' in models_to_train:
        dynamic_lstm_model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
        reset_seeds()

        model_history = dynamic_lstm_model.fit(
            X_train_padded,
            y_train,
            epochs=epochs,
            batch_size=dynamic_batch_size,
            validation_data=(X_val_padded, y_val),
            callbacks=callbacks
        )

        plot_graphs(model_history, "accuracy", 'dynamic_lstm')
        plot_graphs(model_history, "loss", 'dynamic_lstm')

        dynamic_lstm_pred_numeric = dynamic_lstm_model.predict(X_val_padded)
        prediction_result_df = eval_model_preds(dynamic_lstm_pred_numeric, y_val, prediction_result_df, 'dynamic_lstm')

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
    x2 = layers.Dense(len(static_feature_names))(x_i_stat)
    x2 = layers.Dense(len(static_feature_names))(x2)
    x2 = layers.Dense(len(static_feature_names))(x2)
    x2 = layers.Dense(len(static_feature_names))(x2)
    x2 = layers.Dense(64)(x2)
    x2 = layers.Dense(64)(x2)
    x2 = layers.Dense(64)(x2)
    x2 = layers.Dense(64)(x2)
    x2 = Activation('relu')(x2)
    x_stat = Dropout(0.1)(x2)

    x_dot = Dot(axes=1, normalize=True)([x_stat, x_dyn])
    x = layers.Dense(128)(x_dot)
    x = layers.Dense(128)(x)
    x = layers.Dense(128)(x)
    x = layers.Dense(128)(x)
    x_o = Dense(1, activation="sigmoid")(x)

    full_lstm_model = Model([x_i_dyn, x_i_stat], x_o)

    full_lstm_model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

    plot_model(full_lstm_model, f"{plot_folder}/full_lstm_model.png")

    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]

    full_lstm_model.summary()

    if 'full_lstm' in models_to_train:
        full_lstm_hist = full_lstm_model.fit(
            [X_train_padded, X_train_static],
            y_train,
            epochs=epochs,
            batch_size=full_batch_size,
            validation_data=([X_val_padded, X_val_static], y_val),
            callbacks=callbacks
        )

        plot_graphs(full_lstm_hist, "accuracy", 'full_lstm')
        plot_graphs(full_lstm_hist, "loss", 'full_lstm')

        full_lstm_pred_numeric = full_lstm_model.predict([X_val_padded, X_val_static])
        prediction_result_df = eval_model_preds(full_lstm_pred_numeric, y_val, prediction_result_df, 'full_lstm')

    if 'rf' in models_to_train:
        rf = RandomForestClassifier()
        rf.fit(X_train_static, y_train)
        rf_pred = rf.predict(X_val_static)
        prediction_result_df = eval_model_preds(rf_pred, y_val, prediction_result_df, 'rf')

    if '1-NN' in models_to_train:
        one_nn_prediction_model = KNeighborsClassifier(n_neighbors=1)
        one_nn_prediction_model.fit(X_train_padded, y_train)
        one_nn_pred = one_nn_prediction_model.predict(X_val_padded)
        prediction_result_df = eval_model_preds(one_nn_pred, y_val, prediction_result_df, '1-NN')

    if model_to_explain == 'dynamic_lstm':
        y_pred = np.array([1 if pred > 0.5 else 0 for pred in dynamic_lstm_pred_numeric])

        model = dynamic_lstm_model
    if model_to_explain == 'full_lstm':
        y_pred = np.array([1 if pred > 0.5 else 0 for pred in full_lstm_pred_numeric])
        model = full_lstm_model
    if model_to_explain == 'rf':
        y_pred = rf_pred
        model = rf
    if model_to_explain == '1-NN':
        y_pred = one_nn_pred
        model = one_nn_prediction_model

    relevant_y_pred = y_pred[-len(validation_neg):]
    X_pred_negative = X_val_padded[-len(validation_neg):][relevant_y_pred == 0]
    X_pred_negative_static = X_val_neg_static[relevant_y_pred == 0]

    original_event_sequences = tokenizer.sequences_to_texts(X_pred_negative)

    trans_results_delete = pd.read_csv(f'{drg_model_path}/drg_delete/preds', header=None)[relevant_y_pred == 0]
    trans_results_delete_retrieve = pd.read_csv(f'{drg_model_path}/drg_delete_retrieve/preds', header=None)[
        relevant_y_pred == 0]
    trans_results_delete.reset_index(inplace=True, drop=True)
    trans_results_delete_retrieve.reset_index(inplace=True, drop=True)

    for restriction_mode in restriction_modes:
        cf_result_df = pd.DataFrame(columns=["metric", "value"])
        current_output_folder = f"{output_folder}/{restriction_mode}"
        os.mkdir(current_output_folder)

        plot_folder = f"{current_output_folder}/plots"
        os.mkdir(plot_folder)

        # enforce restrictions
        compliant_delete_sequences_idx = list(trans_results_delete_retrieve.index)
        compliant_delete_retrieve_sequences_idx = list(trans_results_delete_retrieve.index)

        if restriction_mode == "cf_restrictions":
            for cf_restriction in cf_restrictions:
                compliant_delete_sequences_idx = cf_restriction \
                    .get_compliant_sequence_indices(
                    list(trans_results_delete[0]),
                    list(validation_neg.diagnoses_text),
                    compliant_delete_sequences_idx
                )

                compliant_delete_retrieve_sequences_idx = cf_restriction \
                    .get_compliant_sequence_indices(
                    list(trans_results_delete_retrieve[0]),
                    list(validation_neg.diagnoses_text),
                    compliant_delete_retrieve_sequences_idx
                )

        nn_model = NearestNeighbors(n_neighbors=1, metric='hamming')
        target_label = 1
        X_target_label = X_train_padded[y_train == target_label]

        nn_model.fit(X_target_label)

        closest = nn_model.kneighbors(X_pred_negative, return_distance=False)
        trans_results_nn = X_target_label[closest[:, 0]]

        X_cf_one_nn = trans_results_nn

        X_cf_delete = tokenizer.texts_to_sequences(trans_results_delete[0][compliant_delete_sequences_idx])
        X_cf_delete_retrieve = tokenizer.texts_to_sequences(
            trans_results_delete_retrieve[0][compliant_delete_retrieve_sequences_idx])

        X_cf_delete_padded = sequence.pad_sequences(X_cf_delete, maxlen=max_seq_length, padding='post')
        X_cf_delete_retrieve_padded = sequence.pad_sequences(X_cf_delete_retrieve, maxlen=max_seq_length, padding='post')

        trans_event_delete = tokenizer.sequences_to_texts(X_cf_delete_padded)
        trans_event_delete_retrieve = tokenizer.sequences_to_texts(X_cf_delete_retrieve_padded)
        trans_event_one_nn = tokenizer.sequences_to_texts(X_cf_one_nn)

        test_size_delete = len(compliant_delete_sequences_idx)
        test_size_delete_retrieve = len(compliant_delete_retrieve_sequences_idx)
        test_size = X_val_neg_static.shape[0]

        if model_to_explain == 'full_lstm':
            fraction_success = np.sum(model.predict(
                [X_cf_delete_padded, X_val_neg_static.loc[compliant_delete_sequences_idx]]) > 0.5) / test_size_delete
        elif model_to_explain == "rf":
            fraction_success = np.sum(model.predict(X_val_neg_static) > 0.5) / test_size
        else:
            fraction_success = np.sum(model.predict(X_cf_delete_padded) > 0.5) / test_size_delete
        print(round(fraction_success, 4))

        cf_result_df = cf_result_df.append(
            {
                'metric': 'frac_delete',
                'value': fraction_success
            },
            ignore_index=True
        )

        if model_to_explain == 'full_lstm':
            fraction_success = np.sum(
                model.predict([X_cf_delete_retrieve_padded, X_val_neg_static.loc[
                    compliant_delete_retrieve_sequences_idx]]) > 0.5) / test_size_delete_retrieve
        elif model_to_explain == "rf":
            fraction_success = np.sum(model.predict(X_val_neg_static) > 0.5) / test_size
        else:
            fraction_success = np.sum(model.predict(X_cf_delete_retrieve_padded) > 0.5) / test_size_delete_retrieve
        print(round(fraction_success, 4))

        cf_result_df = cf_result_df.append(
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

        cf_result_df = cf_result_df.append(
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

        cf_result_df = cf_result_df.append(
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

        cf_result_df = cf_result_df.append(
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

        cf_result_df = cf_result_df.append(
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

        pairwise_bleu_delete = get_pairwise_bleu(
            [original_event_sequences[i] for i in compliant_delete_sequences_idx],
            trans_event_delete
        )
        avg_bleu_delete = sum(pairwise_bleu_delete) / test_size
        print(round(avg_bleu_delete, 4))

        cf_result_df = cf_result_df.append(
            {
                'metric': 'avg_bleu_delete',
                'value': avg_bleu_delete
            },
            ignore_index=True
        )

        pairwise_bleu_delete_retrieve = get_pairwise_bleu(
            [original_event_sequences[i] for i in compliant_delete_retrieve_sequences_idx],
            trans_event_delete_retrieve
        )
        avg_bleu_delete_retrieve = sum(pairwise_bleu_delete_retrieve) / test_size
        print(round(avg_bleu_delete_retrieve, 4))

        cf_result_df = cf_result_df.append(
            {
                'metric': 'avg_bleu_delete_retrieve',
                'value': avg_bleu_delete_retrieve
            },
            ignore_index=True
        )

        pariwise_bleu_one_nn = get_pairwise_bleu(original_event_sequences, trans_event_one_nn)
        avg_bleu_one_nn = sum(pariwise_bleu_one_nn) / test_size
        print(round(avg_bleu_one_nn, 4))

        cf_result_df = cf_result_df.append(
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

        plt.savefig(f"{plot_folder}/histograms_{model_to_explain}.png")

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

        plt.savefig(f"{plot_folder}/difference_plot-{model_to_explain}.png")

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
        cf_result_df.to_csv(f"{current_output_folder}/CF_metrics.csv")
        prediction_result_df.to_csv(f"{current_output_folder}/prediction_metrics.csv")

        for sample_id in sequences_to_plot:
            try:
                print(f"Plotting sequence with ID: {sample_id} ...")

                code_to_name(original_event_sequences[sample_id])

                plot_sequence(
                    code_to_name(original_event_sequences[sample_id]),
                    title=f"ID_{sample_id}-{model_to_explain}-Original_Sequence",
                    output_folder=plot_folder
                )

                if sample_id in compliant_delete_sequences_idx:
                    relevant_id = [k for k, v in enumerate(compliant_delete_sequences_idx) if v == sample_id][0]
                    code_to_name(trans_event_delete[relevant_id])

                    plot_sequence(
                        code_to_name(trans_event_delete[sample_id]),
                        title=f"ID_{sample_id}-{model_to_explain}-DRG-DELETE_Sequence",
                        output_folder=plot_folder
                    )

                if sample_id in compliant_delete_retrieve_sequences_idx:
                    relevant_id = [k for k, v in enumerate(compliant_delete_retrieve_sequences_idx) if v == sample_id][0]
                    code_to_name(trans_event_delete_retrieve[relevant_id])

                    plot_sequence(
                        code_to_name(trans_event_delete_retrieve[relevant_id]),
                        title=f"ID_{sample_id}-{model_to_explain}-DRG_DELETE-RETRIEVE_Sequence",
                        output_folder=plot_folder
                    )

                code_to_name(trans_event_one_nn[sample_id])

                plot_sequence(
                    code_to_name(trans_event_one_nn[sample_id]),
                    title=f"ID_{sample_id}-{model_to_explain}-1-NN_Sequence",
                    output_folder=plot_folder
                )
            except Exception as e:
                print(f"Error for sample {sample_id}: {e}")
