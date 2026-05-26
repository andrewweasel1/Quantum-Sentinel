import os
import json
import datetime
import config

def deploy_champion(booster, feature_list, sector_name, precision):
    """
    Serializes a winning model booster and freezes its precise feature sequence.
    Prepares the model for isolated live-sandbox execution.
    """
    # Ensure the production models directory exists
    os.makedirs(config.PROD_MODELS_DIR, exist_ok=True)
    
    # Create a unique timestamped filename to prevent accidental overwriting
    # and to allow multiple models to compete in the live sandbox.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_sector = str(sector_name).lower().replace(" ", "_")
    base_filename = f"{config.RUN_MODE.lower()}_{clean_sector}_{timestamp}"
    
    model_filepath = os.path.join(config.PROD_MODELS_DIR, f"{base_filename}.json")
    features_filepath = os.path.join(config.PROD_MODELS_DIR, f"{base_filename}_features.json")
    
    # 1. Freeze the XGBoost Model Graph
    # Using the .json extension saves the model in XGBoost's universal format, 
    # preserving the internal tree structures and attributes.
    booster.save_model(model_filepath)
    
    # 2. Freeze the Feature Manifest
    # This strictly locks the input coordinate order for the live_trader.py script
    # to enforce schema alignment before passing live data to the GPU.
    with open(features_filepath, "w") as f:
        json.dump(feature_list, f)
        
    print(f"[{sector_name}] Champion Deployed to Production Pool! Precision: {precision*100:.2f}% | ID: {base_filename}")