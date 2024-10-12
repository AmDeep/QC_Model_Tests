# Web app
import streamlit as st
from annotated_text import annotated_text
from millify import millify
import plotly.express as px
# OS and file management
import os
import pickle
from PIL import Image
import zipfile
# Data analysis
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# General options
im = Image.open("favicon.ico")
st.set_page_config(
    page_title="AMPredST",
    page_icon=im,
    layout="wide",
)

# Attach customized ccs style
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Function to unzip the model
@st.experimental_singleton
def unzip_model(zip_file_name):
    # opening Zip using 'with' keyword in read mode
    with zipfile.ZipFile(zip_file_name, 'r') as file:
        # printing all the information of archive file contents using 'printdir' method
        print(file.printdir())
        # extracting the files using 'extracall' method
        print('Extracting all files...')
        file.extractall()
        print('Done!') # check your directory of zip file to see the extracted files

# Function load the best ML model
@st.experimental_singleton 
def load_model(model_file):
    with open(model_file, 'rb') as f_in:
        model = pickle.load(f_in)
    return model 

# Add a title and info about the app
st.title('AMPredST: Antimicrobial peptides prediction streamlit app')
"""
thank you
"""

with st.expander('About this app'):
    st.write('''
    [Antimicrobial peptides](https://en.wikipedia.org/wiki/Antimicrobial_peptides) (AMPs) are small bioactive drugs, commonly with fewer than 50 amino acids, 
    which have appeared as promising compounds to control infectious disease caused by multi-drug resistant bacteria or superbugs. According to the World Health 
    Organization, suberbugs are one of the [top ten global public health threats facing humanity in this century](https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance), 
    so it is important to search for AMPs that combat multi-drug resistant bacteria.)

# Set the session state to store the peptide sequence
if 'peptide_input' not in st.session_state:
    st.session_state.peptide_input = ''

# Input peptide
st.sidebar.subheader('Input peptide sequence')

def insert_active_peptide_example():
    st.session_state.peptide_input = 'LLNQELLLNPTHQIYPVA'

def insert_inactive_peptide_example():
    st.session_state.peptide_input = 'KSAGYDVGLAGNIGNSLALQVAETPHEYYV'

def clear_peptide():
    st.session_state.peptide_input = ''

peptide_seq = st.sidebar.text_input('Enter peptide sequence', st.session_state.peptide_input, key='peptide_input', help='Be sure to enter a valid peptide sequence')

st.sidebar.button('Example of an active AMP', on_click=insert_active_peptide_example)
st.sidebar.button('Example of an inactive peptide', on_click=insert_inactive_peptide_example)
st.sidebar.button('Clear input', on_click=clear_peptide)

st.sidebar.header('Code availability')







if st.session_state.peptide_input == '':
    st.subheader('Welcome to the app!')
    st.info('Enter peptide sequence in the sidebar to proceed', icon='👈')
else:
    st.subheader('⚛️ Input peptide:')
    st.info(peptide_seq)

if st.session_state.peptide_input != '':

    # General properties of the peptide
    st.subheader('General properties of the peptide')
    analysed_seq = ProteinAnalysis(peptide_seq)
    mol_weight = analysed_seq.molecular_weight()
    aromaticity = analysed_seq.aromaticity()
    instability_index = analysed_seq.instability_index()
    isoelectric_point = analysed_seq.isoelectric_point()
    charge_at_pH = analysed_seq.charge_at_pH(7.0)
    gravy = analysed_seq.gravy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Molecular weight (kDa)", millify(mol_weight/1000, precision=3))
    col2.metric("Aromaticity", millify(aromaticity, precision=3))
    col3.metric("Isoelectric point", millify(isoelectric_point, precision=3))

    col4, col5, col6 = st.columns(3)
    col4.metric("Instability index", millify(instability_index, precision=3))
    col5.metric("Charge at pH 7", millify(charge_at_pH, precision=3))
    col6.metric("Gravy index", millify(gravy, precision=3))

    # Bar plot of the aminoacid composition
    count_amino_acids = analysed_seq.count_amino_acids()

    st.subheader('Aminoacid composition')

    df_amino_acids =  pd.DataFrame.from_dict(count_amino_acids, orient='index', columns=['count'])
    df_amino_acids['aminoacid'] = df_amino_acids.index
    df_amino_acids['count'] = df_amino_acids['count'].astype(int)

    plot = px.bar(df_amino_acids, y='count', x='aminoacid',
              text_auto='.2s', labels={
                      "count": "Count",
                      "aminoacid": "Aminoacid"
                  })
    plot.update_traces(textfont_size=12, textangle=0, textposition="outside", showlegend=False)
    st.plotly_chart(plot)

    # Calculate Amino acid composition feature from the AMP
    amino_acids_percent = analysed_seq.get_amino_acids_percent()

    amino_acids_percent = {key: value*100 for (key, value) in amino_acids_percent.items()}

    df_amino_acids_percent =  pd.DataFrame.from_dict(amino_acids_percent, orient='index', columns=['frequencies']).T

    st.subheader('Molecular descriptors of the peptide (Amimoacid frequencies)')

    st.write(df_amino_acids_percent)

    # Load the best ML model
    # assigning filename to a variable
    zip_file_name = 'ExtraTreesClassifier_maxdepth50_nestimators200.zip'

    # unzip the file with the defined function
    unzip_model(zip_file_name)

    # Load the model
    model_file = 'ExtraTreesClassifier_maxdepth50_nestimators200.bin'
    model = load_model(model_file)
    
    # Predict the AMP activity
    y_pred = model.predict_proba(df_amino_acids_percent)[0, 1]
    active = y_pred >= 0.5

    # Print results
    st.subheader('Prediction of the AMP activity using the machine learning model')
    if active == True:
      annotated_text(
      ("The input molecule is an active AMP", "", "#b6d7a8"))
      annotated_text(
      "Probability of antimicrobial activity: ",
      (f"{y_pred}","", "#b6d7a8"))
    else:
      annotated_text(
      ("The input molecule is not an active AMP", "", "#ea9999"))
      annotated_text(
      "Probability of antimicrobial activity: ",
      (f"{y_pred}","", "#ea9999"))


