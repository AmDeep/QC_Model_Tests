"""
Program to build bioactivity prediction tool streamlit app
"""

# importing necessary libraries
import streamlit as st
import pandas as pd
from PIL import Image
from padelpy import padeldescriptor
import glob
import base64
import pickle

# molecular descriptor calculator
def desc_calc():
    """
    function to initiate PaDEL software via padelpy library to calculate molecular descriptors
    :return: calculated descriptors of input molecules
    """
    # list and sort fingerprint XML files to initiate padelpy/padel software
    xml_files = glob.glob('*.xml')
    xml_files.sort()

    # create list of "keys" for all padel fingerprint types to use for dictionary
    fp_list = ['AtomPairs2DCount',
               'AtomPairs2D',
               'EState',
               'CDKextended',
               'CDK',
               'CDKgraphonly',
               'KlekotaRothCount',
               'KlekotaRoth',
               'MACCS',
               'PubChem',
               'SubstructureCount',
               'Substructure']

    # create dictionary of keys and corresponding values (xml files)
    fp = dict(zip(fp_list, xml_files))

    # initiate padelpy and set fingerprint to "PubChem" type to calculate PubChem fingerprints
    fingerprint = 'PubChem'
    fingerprint_output_file = ''.join([fingerprint,'.csv'])     # output: Pubchem.csv
    fingerprint_descriptortypes = fp[fingerprint]

    padeldescriptor(mol_dir='molecule.smi',
                    d_file=fingerprint_output_file,  # 'PubChem.csv'
                    descriptortypes=fingerprint_descriptortypes,     # descriptortypes='PubChem.xml'
                    detectaromaticity=True,
                    standardizenitro=True,
                    standardizetautomers=True,
                    threads=2,
                    removesalt=True,
                    log=True,
                    fingerprints=True)


# file download
def filedownload(df):
    """
    function to enable users to download prediction results from webapp
    :param df:dataframe of predicted pIC50 values of input molecules
    :return: csv file
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

# model building
def build_model(input_data):
    """
    function to read in Random Forest model built and saved as a pickle object. Function then uses model to predict
    pIC50 values from the molecular descriptors calculated from the input molecules.
    :param input_data: dataframe containing molecular descriptors
    :return:
    """
    # reads in saved Random Forest model
    load_model = pickle.load(open('model.pkl', 'rb'))
    # apply model to make predictions
    prediction = load_model.predict(input_data)
    st.header('**Prediction output**')
    prediction_output = pd.Series(prediction, name='pIC50')
    molecule_name = pd.Series(load_data[1], name='molecule_name')
    df = pd.concat([molecule_name, prediction_output], axis=1)
    st.write(df)
    st.markdown(filedownload(df), unsafe_allow_html=True)

# logo image for app
image = Image.open('logo.png')
st.image(image, use_column_width=True)

# formatting streamlit app
# page title
st.markdown("""
# Bioactivity Prediction App for target protein, Hepatocyte Growth Factor Receptor (HGFR)

Abnormal expression of hepatocyte growth factor receptor (HGFR) has been associated with tumors with high cell proliferation rate, 
cell dissociation, tumor aggressiveness, and invasiveness. Literature has supported it as being a promising target in cancer therapy, however the protein's 
characteristic of therapeutic resistance is still a barrier for developing effective drugs that can inhibit it. Discovery of novel compounds/molecules with 
promising bioactivity with HGFR is needed. 

This app allows users to determine if a molecule of interest has the potential to be a drug candidate by predicting the 
bioactivity towards inhibiting the HGFR. The app accepts .txt files containing the canonical simplified molecular-input line-entry system (SMILES) 
notation of the molecule(s) of interest and respective ChEMBLID. Results will return predicted pIC50 value(s) of the molecule(s)
of interest.

A note on predicted pIC50 values: the prediction model was trained with consideration of compounds with pIC50 values >6 as **active** while values <5 as **inactive**.


""")

# sidebar
with st.sidebar.header('Upload your molecule .txt file'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
    st.sidebar.markdown("""
[Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
""")

if st.sidebar.button('Predict'):
    load_data = pd.read_table(uploaded_file, sep=' ', header=None)
    load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

    st.header('**Original input data**')
    st.write(load_data)

    with st.spinner("Calculating descriptors..."):
        desc_calc()

    # Read in calculated descriptors and display the dataframe
    st.header('**Calculated molecular descriptors**')
    desc = pd.read_csv('PubChem.csv')
    st.write(desc)
    st.write(desc.shape)

    # Read descriptor list used in previously built model
    st.header('**Subset of descriptors from built model**')
    Xlist = list(pd.read_csv('descriptor_list.csv').columns)
    desc_subset = desc[Xlist]
    st.write(desc_subset)
    st.write(desc_subset.shape)

    # Apply trained model to make prediction on query compounds
    build_model(desc_subset)
else:
    st.info('Upload input data in the sidebar to start!')
