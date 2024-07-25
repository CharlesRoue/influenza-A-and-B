import time
import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

model_A = CatBoostClassifier().load_model("./model/model_A")
model_B = CatBoostClassifier().load_model("./model/model_B")

active_tab = st.session_state.get("active_tab", "Automatic entry")


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")


def data_validation(df):
    df_to_num = df.apply(pd.to_numeric, errors="coerce")
    has_null_values = df_to_num.isnull().values.any()
    if has_null_values:
        return False
    else:
        return True


st.set_page_config("Streamlit Components Hub", "🎪", layout="wide")
st.image("./resources/Ai Lab.png", width=500)

col_left, col_right = st.columns(2)

submit_dataframe_blood_param = pd.DataFrame()


with col_left:
    st.markdown("### 1. Entering patient information.")
    auto_entry, manu_entry, batch_entry = st.tabs(
        ["Automatic entry", "Manual entry", "batch entry"]
    )
    with auto_entry:
        st.text_input(label="Please enter the patient ID:")  # 添加key参数以便跟踪输入
        col_enquiry, col_submit = st.columns([1, 1])
        with col_enquiry:
            if st.button("Enquiry patient", use_container_width=True):
                # Need to connect to the LIS system for real-time inspection results!
                st.write("Please connect to the HIS system first......")
        with col_submit:
            submit_dataframe_button = st.button(
                "Submit", key="auto_submit", use_container_width=True, type="primary"
            )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            data = st.text_input(label="Date:", key="Date_auto")
            age = st.text_input(label="Age:", key="Age_auto")
            wbc = st.text_input(label="WBC:", key="WBC_auto")
            mcv = st.text_input(label="MCV:", key="MCV_auto")
            pdw = st.text_input(label="PDW:", key="PDW_auto")
            lymph0 = st.text_input(label="LYMPH%:", key="LYMPH_auto0")
            neut1 = st.text_input(label="NEUT#:", key="NEUT_auto1")
            baso1 = st.text_input(label="BASO#:", key="BASO_auto1")

        with col2:
            id = st.text_input(label="Patient ID:", key="Patient_ID_auto")
            department = st.text_input(label="Department:", key="Department_auto")
            rbc = st.text_input(label="RBC:", key="RBC_auto")
            mch = st.text_input(label="MCH:", key="MCH_auto")
            mpv = st.text_input(label="MPV:", key="MPV_auto")
            mono0 = st.text_input(label="MONO%:", key="MONO_auto0")
            lymph1 = st.text_input(label="LYMPH#:", key="LYMPH_auto1")
            rdw_cv = st.text_input(label="RDW-CV:", key="RDW_CV_auto")

        with col3:
            name = st.text_input(label="Name:", key="Name_auto")
            diagnostic = st.text_input(label="Diagnostic:", key="Diagnostic_auto")
            hb = st.text_input(label="HB:", key="HB_auto")
            mchc = st.text_input(label="MCHC:", key="MCHC_auto")
            pct = st.text_input(label="PCT:", key="PCT_auto")
            eos = st.text_input(label="EOS%:", key="EOS_auto0")
            mono1 = st.text_input(label="MONO#:", key="MONO_auto1")
            p_lcr = st.text_input(label="P-LCR:", key="P_LCR_auto")

        with col4:
            sex = st.text_input(label="Sex:", key="Sex_auto")
            resp_doctor = st.text_input(
                label="Responsible doctor:", key="Responsible_doctor_auto"
            )
            hct = st.text_input(label="HCT:", key="HCT_auto")
            plt = st.text_input(label="PLT:", key="PLT_auto")
            neut0 = st.text_input(label="NEUT%:", key="NEUT_auto0")
            baso0 = st.text_input(label="BASO%:", key="BASO_auto0")
            eos1 = st.text_input(label="EOS#:", key="EOS_auto1")

        submit_dataframe_patient_info = pd.DataFrame(
            {
                "Date": [st.session_state["Date_auto"]],
                "Patient ID": [st.session_state["Patient_ID_auto"]],
                "Name": [st.session_state["Name_auto"]],
                "Sex": [st.session_state["Sex_auto"]],
                "Age": [st.session_state["Age_auto"]],
                "Department": [st.session_state["Department_auto"]],
                "Diagnostic": [st.session_state["Diagnostic_auto"]],
                "Responsible doctor": [st.session_state["Responsible_doctor_auto"]],
            },
            index=[0],
        )

        submit_dataframe_blood_param = pd.DataFrame(
            [
                {
                    "WBC": st.session_state["WBC_auto"],
                    "RBC": st.session_state["RBC_auto"],
                    "HB": st.session_state["HB_auto"],
                    "HCT": st.session_state["HCT_auto"],
                    "MCV": st.session_state["MCV_auto"],
                    "MCH": st.session_state["MCH_auto"],
                    "MCHC": st.session_state["MCHC_auto"],
                    "PLT": st.session_state["PLT_auto"],
                    "PDW": st.session_state["PDW_auto"],
                    "MPV": st.session_state["MPV_auto"],
                    "PCT": st.session_state["PCT_auto"],
                    "NEUT%": st.session_state["NEUT_auto0"],
                    "LYMPH%": st.session_state["LYMPH_auto0"],
                    "MONO%": st.session_state["MONO_auto0"],
                    "EOS%": st.session_state["EOS_auto0"],
                    "BASO%": st.session_state["BASO_auto0"],
                    "NEUT#": st.session_state["NEUT_auto1"],
                    "LYMPH#": st.session_state["LYMPH_auto1"],
                    "MONO#": st.session_state["MONO_auto1"],
                    "EOS#": st.session_state["EOS_auto1"],
                    "BASO#": st.session_state["BASO_auto1"],
                    "RDW-CV": st.session_state["RDW_CV_auto"],
                    "P-LCR": st.session_state["P_LCR_auto"],
                }
            ],
            index=[0],
        )
        if submit_dataframe_button:
            st.write(submit_dataframe_blood_param)

    with manu_entry:
        submit_dataframe_button = st.button("Submit", key="manu_submit", type="primary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.date_input(label="Date:", key="Date_manu")
            data = st.text_input(label="Age:", key="Age_manu", value="26")
            wbc = st.text_input(label="WBC:", key="WBC_manu", value="6.66")
            mcv = st.text_input(label="MCV:", key="MCV_manu", value="88")
            pdw = st.text_input(label="PDW:", key="PDW_manu", value="10.2")
            lymph0 = st.text_input(label="LYMPH%:", key="LYMPH_manu0", value="9")
            neut1 = st.text_input(label="NEUT#:", key="NEUT_manu1", value="5.54")
            baso1 = st.text_input(label="BASO#:", key="BASO_manu1", value="0.02")

        with col2:
            id = st.text_input(
                label="Patient ID:", key="Patient_ID_manu", value="64239414"
            )
            department = st.text_input(
                label="Department:", key="Department_manu", value="emergency"
            )
            rbc = st.text_input(label="RBC:", key="RBC_manu", value="4.49")
            mch = st.text_input(label="MCH:", key="MCH_manu", value="28.7")
            mpv = st.text_input(label="MPV:", key="MPV_manu", value="9.7")
            mono0 = st.text_input(label="MONO%:", key="MONO_manu0", value="7.2")
            lymph1 = st.text_input(label="LYMPH#:", key="LYMPH_manu1", value="0.6")
            rdw_cv = st.text_input(label="RDW-CV:", key="RDW_CV_manu", value="12.8")

        with col3:
            name = st.text_input(label="Name:", key="Name_manu", value="MICH****")
            diagnostic = st.text_input(
                label="Diagnostic:", key="Diagnostic_manu", value="fever"
            )
            hb = st.text_input(label="HB:", key="HB_manu", value="129")
            mchc = st.text_input(label="MCHC:", key="MCHC_manu", value="326.6")
            pct = st.text_input(label="PCT:", key="PCT_manu", value="0.28")
            eos = st.text_input(label="EOS%:", key="EOS_manu0", value="0.3")
            mono1 = st.text_input(label="MONO#:", key="MONO_manu1", value="0.48")
            p_lcr = st.text_input(label="P-LCR:", key="P_LCR_manu", value="21.7")

        with col4:
            sex = st.text_input(label="Sex:", key="Sex_manu", value="female")
            resp_doctor = st.text_input(
                label="Responsible doctor:",
                key="Responsible_doctor_manu",
                value="Qiang Wang",
            )
            hct = st.text_input(label="HCT:", key="HCT_manu", value="39.5")
            plt = st.text_input(label="PLT:", key="PLT_manu", value="287")
            neut0 = st.text_input(label="NEUT%:", key="NEUT_manu0", value="83.2")
            baso0 = st.text_input(label="BASO%:", key="BASO_manu0", value="0.3")
            eos1 = st.text_input(label="EOS#:", key="EOS_manu1", value="0.02")

        submit_dataframe_patient_info = pd.DataFrame(
            {
                "Date": [st.session_state["Date_manu"]],
                "Patient ID": [st.session_state["Patient_ID_manu"]],
                "Name": [st.session_state["Name_manu"]],
                "Sex": [st.session_state["Sex_manu"]],
                "Age": [st.session_state["Age_manu"]],
                "Department": [st.session_state["Department_manu"]],
                "Diagnostic": [st.session_state["Diagnostic_manu"]],
                "Responsible doctor": [st.session_state["Responsible_doctor_manu"]],
            },
            index=[0],
        )

        submit_dataframe_blood_param = pd.DataFrame(
            [
                {
                    "WBC": st.session_state["WBC_manu"],
                    "RBC": st.session_state["RBC_manu"],
                    "HB": st.session_state["HB_manu"],
                    "HCT": st.session_state["HCT_manu"],
                    "MCV": st.session_state["MCV_manu"],
                    "MCH": st.session_state["MCH_manu"],
                    "MCHC": st.session_state["MCHC_manu"],
                    "PLT": st.session_state["PLT_manu"],
                    "PDW": st.session_state["PDW_manu"],
                    "MPV": st.session_state["MPV_manu"],
                    "PCT": st.session_state["PCT_manu"],
                    "NEUT%": st.session_state["NEUT_manu0"],
                    "LYMPH%": st.session_state["LYMPH_manu0"],
                    "MONO%": st.session_state["MONO_manu0"],
                    "EOS%": st.session_state["EOS_manu0"],
                    "BASO%": st.session_state["BASO_manu0"],
                    "NEUT#": st.session_state["NEUT_manu1"],
                    "LYMPH#": st.session_state["LYMPH_manu1"],
                    "MONO#": st.session_state["MONO_manu1"],
                    "EOS#": st.session_state["EOS_manu1"],
                    "BASO#": st.session_state["BASO_manu1"],
                    "RDW-CV": st.session_state["RDW_CV_manu"],
                    "P-LCR": st.session_state["P_LCR_manu"],
                }
            ],
            index=[0],
        )
        if submit_dataframe_button:
            st.write(submit_dataframe_blood_param)

    with batch_entry:
        temp_csv = convert_df(
            pd.DataFrame(
                columns=[
                    "Date",
                    "Patient ID",
                    "Name",
                    "Sex",
                    "Age",
                    "Department",
                    "Diagnostic",
                    "Responsible doctor",
                    "WBC",
                    "RBC",
                    "HB",
                    "HCT",
                    "MCV",
                    "MCH",
                    "MCHC",
                    "PLT",
                    "PDW",
                    "MPV",
                    "PCT",
                    "NEUT%",
                    "LYMPH%",
                    "MONO%",
                    "EOS%",
                    "BASO%",
                    "NEUT#",
                    "LYMPH#",
                    "MONO#",
                    "EOS#",
                    "BASO#",
                    "RDW-CV",
                    "P-LCR",
                ]
            )
        )
        st.download_button(
            "Click to download the batch entry template file.",
            data=temp_csv,
            file_name="temp.csv",
            mime="text/csv",
        )
        uploaded_file = st.file_uploader(
            "Choose a CSV file", accept_multiple_files=False
        )
        if uploaded_file is not None:
            dataframe = pd.read_csv(uploaded_file, encoding="GBK")
            st.write(dataframe)
            submit_dataframe_patient_info = dataframe.iloc[:, :8]
            submit_dataframe_blood_param = dataframe.iloc[:, 8:]
with col_right:
    start_ai_diag = None
    st.markdown("### 2. Automatic diagnosis by artificial intelligence")
    if st.button("Click here to verify data validity", key="upload_data"):
        if submit_dataframe_blood_param.shape[0] == 1:
            st.write(submit_dataframe_patient_info)
            st.write(submit_dataframe_blood_param)
        else:
            st.write(
                pd.concat(
                    [submit_dataframe_patient_info, submit_dataframe_blood_param],
                    axis=1,
                )
            )
        with st.spinner("Validation of data validity in progress..."):
            time.sleep(3)
            if data_validation(submit_dataframe_blood_param):
                st.success("The data is valid and the operation can be continued!")
                start_ai_diag = True
            else:
                st.error("Data error, please re-enter!")
        if start_ai_diag:
            with st.spinner("Auto-diagnostics in progress...."):
                time.sleep(5)
                result_prob_A = model_A.predict_proba(submit_dataframe_blood_param)
                result_prob_B = model_B.predict_proba(submit_dataframe_blood_param)
                result_prob_A = pd.DataFrame(data=result_prob_A)
                result_prob_B = pd.DataFrame(data=result_prob_B)
                # st.write(model_A.predict_proba(submit_dataframe_blood_param))
                # st.write(model_B.predict_proba(submit_dataframe_blood_param))
                # st.write(model_B.predict_proba(submit_dataframe_blood_param).shape[0])

                if model_B.predict_proba(submit_dataframe_blood_param).shape[0] != 1:
                    results = []
                    for i in range(len(result_prob_A)):
                        # st.write(result_prob_A)
                        # time.sleep(300)
                        if result_prob_A.iloc[i, 0] > result_prob_A.iloc[i, 1]:
                            results.append("Negative")
                        else:
                            if result_prob_A.iloc[i, 1] > result_prob_B.iloc[i, 1]:
                                results.append("Influenza A")
                            elif result_prob_B.iloc[i, 1] > result_prob_A.iloc[i, 1]:
                                results.append("Influenza B")
                    results_df = submit_dataframe_patient_info[
                        ["Patient ID", "Name", "Sex", "Age"]
                    ]
                    results_df["Ai Diagnostic"] = results
                    st.dataframe(results_df, use_container_width=True)
                else:
                    results = []
                    if result_prob_A.iloc[0, 0] > result_prob_A.iloc[0, 1]:
                        results.append("Negative")
                    else:
                        if result_prob_A.iloc[0, 1] >= result_prob_B.iloc[0, 1]:
                            results.append("Influenza A")
                        elif result_prob_B.iloc[0, 1] > result_prob_A.iloc[0, 1]:
                            results.append("Influenza B")
                    results_df = submit_dataframe_patient_info[
                        ["Patient ID", "Name", "Sex", "Age"]
                    ]
                    results_df["Ai Diagnostic"] = results
                    st.dataframe(results_df, use_container_width=True)
                # time.sleep(300)
