from flask import Flask, request, jsonify
import pickle
import pandas as pd
from rdkit.Chem import MolFromSmiles
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors  # Import Descriptors here
from xgboost import XGBRegressor

app = Flask(__name__)

def calculate_descriptors(smiles):
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    generated_features = []
    for sm in smiles:
        mol = MolFromSmiles(sm)
        if mol:
            generated_features.append(calc.CalcDescriptors(mol))
        else:
            generated_features.append(None)
    return pd.DataFrame(generated_features, columns=calc.GetDescriptorNames(), index=smiles)

def load_model(model_type):
    if model_type == 'Linear Regression':
        return pickle.load(open('models/solubility_model.pkl', 'rb'))
    elif model_type == 'XGBoost':
        model = XGBRegressor()
        model.load_model("models/xgboost_regressor.json")
        return model

def remove_invalid(smiles):
    valid = [sm for sm in smiles if MolFromSmiles(sm)]
    return valid, len(valid) == len(smiles)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'smiles' not in data:
        return jsonify({"error": "Invalid input! Please provide a list of SMILES."}), 400

    smiles_input = data['smiles']
    if not isinstance(smiles_input, list):
        return jsonify({"error": "SMILES must be a list."}), 400
    
    if not smiles_input:
        return jsonify({"error": "The list of SMILES cannot be empty."}), 400

    model_type = data.get('model', 'XGBoost')  # Default to XGBoost if not specified
    valid_models = ['Linear Regression', 'XGBoost']
    if model_type not in valid_models:
        return jsonify({"error": f"Invalid model type! Choose from {valid_models}."}), 400

    # Remove invalid SMILES
    smiles, all_valid = remove_invalid(smiles_input)
    if not all_valid:
        return jsonify({"message": "Some SMILES are invalid! Showing results for valid SMILES only.", "valid_smiles": smiles}), 400

    try:
        # Calculate molecular descriptors
        X = calculate_descriptors(smiles)

        # Load the model
        trained_model = load_model(model_type)

        # Make predictions
        predictions = trained_model.predict(X)
        preds_df = pd.DataFrame(predictions, columns=['Predicted LogS'], index=smiles)

        return jsonify(preds_df.to_dict(orient='index'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
