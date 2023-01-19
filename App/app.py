# Import Core Pkgs
import streamlit as st 
import altair as alt
import pandas as pd 
import numpy as np 
import joblib 

pipeline_lib = joblib.load(open('models/emotion_classifier_pipe.pkl', 'rb'))

def predict_emotions(docx):
    results = pipeline_lib.predict([docx])
    return results[0] # only the first element in the emotion list

def get_prediction_proba(docx):
    proba = pipeline_lib.predict_proba([docx])
    return proba # dictionary of emotions and probability associated with each

def main():
    st.title("How Are You Feeling?")
    menu = ["Home", 'About']
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home-Emotion In Text")

        with st.form(key='emotion_clf_form'): # Receving text from user
            raw_text = st.text_area("Type Here:") # Store received text
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2= st.columns(2)

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text:")
                st.write(raw_text)

                st.success("Prediction:")
                st.write(prediction)
                st.write("How Correct?: {}%".format(np.max(probability)))

            with col2:
                st.success("Probability of the Preidction:")
                proba_df = pd.DataFrame(probability, columns=pipeline_lib.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["Emotions", "Probaility (%)"]
                st.write(proba_df_clean.sort_values("Probaility (%)", ascending = False).reset_index(drop=True))

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='Emotions', y='Probaility (%)')
                st.altair_chart(fig, use_container_width=True)
                
    else:
        st.subheader("About")
        st.write("This is a simple text emotion classifier app deployed using streamlit made by Jay")

if __name__ == "__main__":
    main()
