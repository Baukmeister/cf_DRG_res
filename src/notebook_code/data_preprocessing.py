from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_data(
        postgres_pw,
        data_path,
        num_tfidf_features,
        tfidf_names
):
    conn = psycopg2.connect(
        database="mimic",
        user='postgres',
        password=postgres_pw,
        host="127.0.0.1",
        port="5432",
        options=f'-c search_path=mimiciii')
    cur = conn.cursor()

    """
    lists of ICD 9 codes (related to heart diseases):

    393-398  Chronic Rheumatic Heart Disease
    410-414  Ischemic Heart Disease
    420-429  Other Forms Of Heart Disease
    """

    print("Getting relevant subjects and their diagnoses ...")
    subjects_and_diagnoses = pd.read_sql(
        """
        select subject_id, string_agg(icd9_code, ' ') AS diagnoses
        from diagnoses_icd 
        group by subject_id;
        """, conn)

    patient_ids = set(subjects_and_diagnoses['subject_id'])

    print("Creating TF-IDF features for diagnoses codes ...")

    vectorizer = TfidfVectorizer(max_features=num_tfidf_features)
    diagnoses_list = [doc if doc is not None else "" for doc in subjects_and_diagnoses.diagnoses.tolist()]
    tfidf_vectors = vectorizer.fit_transform(diagnoses_list).toarray()

    subjects_and_diagnoses = pd.concat([subjects_and_diagnoses, pd.DataFrame(tfidf_vectors)], axis=1)
    subjects_and_diagnoses.columns = ['subject_id', 'diagnoses_text'] + tfidf_names

    print("Getting relevant admission events ...")

    admissions_diff = pd.read_sql(
        """
        SELECT a.subject_id, a.hadm_id,
        ROUND((cast(a.admittime as date)-cast(last_admission_time.max_admittime as date))/365.242,2) AS diff_from_last 
        FROM admissions AS a
        LEFT JOIN
            (SELECT subject_id,  MAX(admittime) AS max_admittime
            FROM admissions
            GROUP BY subject_id
            ) AS last_admission_time
        ON a.subject_id=last_admission_time.subject_id;
        """, conn)

    admissions_last_year = admissions_diff[admissions_diff['diff_from_last'] >= -1]

    hadm_id_set = set(admissions_last_year['hadm_id'])

    print("Getting relevant drug events ...")
    view_name = 'drug_event_view'
    drug_view_exists = pd.read_sql(
        f"select exists(select * from pg_views where viewname = '{view_name}');",
        conn)['exists'][0]

    if not drug_view_exists:
        print("Creating drug_event_view ...")
        cur.execute(
            f"""
            CREATE View {view_name} AS
            SELECT im.subject_id, im.hadm_id, im.starttime, im.itemid, di.abbreviation
            FROM inputevents_mv as im
            JOIN d_items as di
            ON im.itemid=di.itemid;
            """
        )
        conn.commit()

    drug_events = pd.read_sql(
        f"""
        SELECT * from {view_name};
        """, conn
    )

    drug_events_last_year = drug_events[drug_events['hadm_id'].isin(hadm_id_set)]

    drug_events_filtered = drug_events_last_year[drug_events_last_year['subject_id'].isin(patient_ids)]

    drug_events_filtered2 = drug_events_filtered.drop_duplicates()

    itemid_counts = drug_events_filtered2['itemid'].value_counts()

    itemid_counts2 = itemid_counts.reset_index()
    itemid_counts2 = itemid_counts2.rename(columns={"index": "itemid", "itemid": "counts"})

    itemid_counts2['proportion'] = itemid_counts2['counts'] / sum(itemid_counts2['counts'])

    itemid_counts3 = itemid_counts2[(itemid_counts2['proportion'] <= 0.041) & (itemid_counts2['counts'] >= 5)]

    itemid_set = set(itemid_counts3['itemid'])

    drug_events_filtered3 = drug_events_filtered2[drug_events_filtered2['itemid'].isin(itemid_set)]

    drug_events_only = drug_events_filtered3.groupby(by='subject_id').apply(lambda x: x.sort_values('starttime'))[
        'itemid'].reset_index(level=[1], drop=True)

    drug_events_by_patient = drug_events_only.groupby(by='subject_id').apply(list)

    drug_events_by_patient2 = drug_events_by_patient.reset_index()

    drug_events_by_patient2['count'] = [len(events) for events in drug_events_by_patient2['itemid']]

    drug_events_by_patient2['count'].describe()

    drug_events_by_patient3 = drug_events_by_patient2[
        drug_events_by_patient2['count'].apply(lambda x: True if x >= 3 and x <= 50 else False)]

    print("Getting relevant procedure codes ...")

    view_name = 'procedure_code_view'
    procedure_code_view_exists = pd.read_sql(
        f"select exists(select * from pg_views where viewname = '{view_name}');",
        conn)['exists'][0]

    if not procedure_code_view_exists:
        print("Creating procedure code view ...")
        cur.execute(
            f"""
                  CREATE View {view_name} AS
                      SELECT a.admittime, procedures.* 
                        FROM admissions AS a
                        RIGHT JOIN
                        (SELECT pi.subject_id, pi.hadm_id, pi.seq_num, pi.icd9_code, dip.short_title
                        FROM procedures_icd AS pi
                        JOIN d_icd_procedures AS dip
                        ON pi.icd9_code=dip.icd9_code) AS procedures
                        ON a.hadm_id=procedures.hadm_id;
                  """
        )
        conn.commit()

    procedure_codes = pd.read_sql(
        f"""
        SELECT * from {view_name};
        """, conn)

    procedure_codes_last_year = procedure_codes[procedure_codes['hadm_id'].isin(hadm_id_set)]

    procedure_codes_filtered = procedure_codes_last_year[
        procedure_codes_last_year['subject_id'].isin(patient_ids)]

    procedure_codes_filtered2 = procedure_codes_filtered.drop(['short_title', 'hadm_id'], axis=1)

    procedure_codes_filtered3 = procedure_codes_filtered2.groupby(by='subject_id').apply(
        lambda x: x.sort_values(['admittime', 'seq_num']))

    procedure_codes_filtered4 = procedure_codes_filtered3.reset_index(level=[0, 1], drop=True)

    procedures_by_patient = procedure_codes_filtered4.groupby(by='subject_id', axis=0)['icd9_code'].apply(list)

    procedures_by_patient2 = procedures_by_patient.reset_index()

    procedures_by_patient2['count'] = [len(codes) for codes in procedures_by_patient2['icd9_code']]

    procedures_by_patient2['count'].describe()

    drug_events_procedures_merged = pd.merge(drug_events_by_patient3, procedures_by_patient2, how='inner',
                                             on='subject_id')

    drug_events_procedures_merged['total_count'] = drug_events_procedures_merged['count_x'] + \
                                                   drug_events_procedures_merged['count_y']

    drug_events_procedures_merged['total_count'].describe()

    drug_events_procedures_merged2 = drug_events_procedures_merged[drug_events_procedures_merged['icd9_code'].notna()]

    drug_events_procedures_merged3 = drug_events_procedures_merged2.drop(['count_x', 'count_y', 'total_count'], axis=1)

    drug_events_procedures_merged4 = drug_events_procedures_merged3.rename(
        columns={"icd9_code": "procedure_codes", "itemid": "drug_events"})

    print("Getting survival data ...")

    survival_subject_ids = pd.read_sql(
        """
        SELECT subject_id FROM patients
        WHERE expire_flag=0;
        """, conn)

    survival_id_set = set(survival_subject_ids['subject_id'])

    drug_events_procedures_merged4['survival'] = [1 if idx in survival_id_set else 0 for idx in
                                                  drug_events_procedures_merged4['subject_id']]

    print("Getting static patient data ...")

    patients = pd.read_sql(
        """
    SELECT ie.subject_id, ie.hadm_id, ie.icustay_id

    -- patient level factors
    , pat.gender, pat.dod

    -- hospital level factors
    , adm.admittime, adm.dischtime
    , ROUND( (CAST(EXTRACT(epoch FROM adm.dischtime - adm.admittime)/(60*60*24) AS numeric)), 4) AS los_hospital
    , ROUND( (CAST(EXTRACT(epoch FROM adm.admittime - pat.dob)/(60*60*24*365.242) AS numeric)), 4) AS age
    , adm.ethnicity
    , case when ethnicity in
      (
           'WHITE' --  40996
         , 'WHITE - RUSSIAN' --    164
         , 'WHITE - OTHER EUROPEAN' --     81
         , 'WHITE - BRAZILIAN' --     59
         , 'WHITE - EASTERN EUROPEAN' --     25
      ) then 'white'
      when ethnicity in
      (
          'BLACK/AFRICAN AMERICAN' --   5440
        , 'BLACK/CAPE VERDEAN' --    200
        , 'BLACK/HAITIAN' --    101
        , 'BLACK/AFRICAN' --     44
        , 'CARIBBEAN ISLAND' --      9
      ) then 'black'
      when ethnicity in
        (
          'HISPANIC OR LATINO' --   1696
        , 'HISPANIC/LATINO - PUERTO RICAN' --    232
        , 'HISPANIC/LATINO - DOMINICAN' --     78
        , 'HISPANIC/LATINO - GUATEMALAN' --     40
        , 'HISPANIC/LATINO - CUBAN' --     24
        , 'HISPANIC/LATINO - SALVADORAN' --     19
        , 'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)' --     13
        , 'HISPANIC/LATINO - MEXICAN' --     13
        , 'HISPANIC/LATINO - COLOMBIAN' --      9
        , 'HISPANIC/LATINO - HONDURAN' --      4
      ) then 'hispanic'
      when ethnicity in
      (
          'ASIAN' --   1509
        , 'ASIAN - CHINESE' --    277
        , 'ASIAN - ASIAN INDIAN' --     85
        , 'ASIAN - VIETNAMESE' --     53
        , 'ASIAN - FILIPINO' --     25
        , 'ASIAN - CAMBODIAN' --     17
        , 'ASIAN - OTHER' --     17
        , 'ASIAN - KOREAN' --     13
        , 'ASIAN - JAPANESE' --      7
        , 'ASIAN - THAI' --      4
      ) then 'asian'
      when ethnicity in
      (
           'AMERICAN INDIAN/ALASKA NATIVE' --     51
         , 'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE' --      3
      ) then 'native'
      when ethnicity in
      (
          'UNKNOWN/NOT SPECIFIED' --   4523
        , 'UNABLE TO OBTAIN' --    814
        , 'PATIENT DECLINED TO ANSWER' --    559
      ) then 'unknown'
      else 'other' end as ethnicity_grouped
      -- , 'OTHER' --   1512
      -- , 'MULTI RACE ETHNICITY' --    130
      -- , 'PORTUGUESE' --     61
      -- , 'MIDDLE EASTERN' --     43
      -- , 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER' --     18
      -- , 'SOUTH AMERICAN' --      8

    , adm.admission_type
    , adm.hospital_expire_flag
    , DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) AS hospstay_seq
    , CASE
        WHEN DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) = 1 THEN True
        ELSE False END AS first_hosp_stay

    -- icu level factors
    , ie.intime, ie.outtime
    , ROUND( (CAST(EXTRACT(epoch FROM ie.outtime - ie.intime)/(60*60*24) AS numeric)), 4) AS los_icu
    , DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) AS icustay_seq

    -- first ICU stay *for the current hospitalization*
    , CASE
        WHEN DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) = 1 THEN True
        ELSE False END AS first_icu_stay

    FROM icustays ie
    INNER JOIN admissions adm
        ON ie.hadm_id = adm.hadm_id
    INNER JOIN patients pat
        ON ie.subject_id = pat.subject_id
    ORDER BY ie.subject_id, adm.admittime, ie.intime;
        """, conn)

    print("Merging all of the data ...")

    patients['age'] = pd.to_numeric([95 if age >= 300 else age for age in patients['age']])
    patients['gender'], _ = pd.factorize(patients['gender'])
    patients['ethnicity'], _ = pd.factorize(patients['ethnicity'])
    patients['admission_type'], _ = pd.factorize(patients['admission_type'])
    patients['first_hosp_stay'], _ = pd.factorize(patients['first_hosp_stay'])
    patients['first_icu_stay'], _ = pd.factorize(patients['first_icu_stay'])
    patients = patients.merge(subjects_and_diagnoses, on="subject_id", how='left')

    final_merged = drug_events_procedures_merged4.copy().merge(patients, on="subject_id", how='left')
    final_merged['drug_events'] = final_merged['drug_events'].apply(lambda x: ' '.join(str(i) for i in x))
    final_merged['procedure_codes'] = final_merged['procedure_codes'].apply(lambda x: ' '.join(x))
    final_merged['events'] = final_merged['drug_events'] + " " + final_merged['procedure_codes']
    final_merged.drop(columns=['subject_id', 'drug_events', 'procedure_codes'], inplace=True)

    neg_data = final_merged[final_merged['survival'] == 0].drop(columns=['survival'])
    pos_data = final_merged[final_merged['survival'] == 1].drop(columns=['survival'])

    validation_neg = neg_data.sample(n=200, random_state=3)
    train_neg = neg_data.drop(validation_neg.index)

    print("Exporting the data ...")

    train_neg.to_csv(path_or_buf=f'./{data_path}/train_neg.txt', index=False)
    validation_neg.to_csv(path_or_buf=f'./{data_path}/validation_neg.txt', index=False)

    validation_pos = pos_data.sample(n=200, random_state=3)
    train_pos = pos_data.drop(validation_pos.index)

    train_pos.to_csv(path_or_buf=f'./{data_path}/train_pos.txt', index=False)
    validation_pos.to_csv(path_or_buf=f'./{data_path}/validation_pos.txt', index=False)

    train_all = pd.concat([train_pos, train_neg])
    train_all.to_csv(path_or_buf=f'./{data_path}/train_all.txt', index=False)

    # store pure event sequences for DRG framework
    train_neg["events"].to_csv(path_or_buf=f'./{data_path}/train_neg_events.txt', index=False, header=False)
    validation_neg["events"].to_csv(path_or_buf=f'./{data_path}/validation_neg_events.txt', index=False, header=False)

    train_pos["events"].to_csv(path_or_buf=f'./{data_path}/train_pos_events.txt', index=False, header=False)
    validation_pos["events"].to_csv(path_or_buf=f'./{data_path}/validation_pos_events.txt', index=False, header=False)

    train_all["events"].to_csv(path_or_buf=f'./{data_path}/train_all_events.txt', index=False, header=False)
