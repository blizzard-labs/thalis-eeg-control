import os
from random import sample
import subprocess
import argparse
import string
import sys
import unicorneeg.stream as ulsl
from unicorneeg.clean import EEGPreprocessor, create_preprocessing_callback

def collect_data(background_prcs=False, graphing=True, duration=0, run_name=""):
    
    def handle_processed(sample):
        #print(f"Processed sample at t={sample['Time']}")
        pass

    preprocessor = EEGPreprocessor()
    
    callback = create_preprocessing_callback(
        preprocessor=preprocessor,
        output_callback=handle_processed,
        output_band='combined'  # or 'alpha', 'beta'
    )

    config = ulsl.EEGStreamConfig(
        use_background_thread=background_prcs,
        enable_graphing=graphing,
        save_duration_seconds=duration,
        csv_path=os.path.join("data/streamlogs", run_name + "_eeg_data.csv")
    )
    
    stream = ulsl.EEGStream(config)
    stream.on_sample(callback)
    
    usrconfirm = input("Press Enter to start data collection...")
    
    print("Connecting to EEG stream...")
    try:
        stream.connect()
        print("Connected! Starting data collection...")
        stream.start()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Make sure the Unicorn EEG device is streaming via LSL.")
        sys.exit(1)


if __name__ == "__main__":
    header = '''
================================================================================================================
 ________ _____   __   _         __    ___               __  __       __  _       _____          __           __
/_  __/ // / _ | / /  (_)__     / /   / _ \_______  ___ / /_/ /  ___ / /_(_)___  / ___/__  ___  / /________  / /
 / / / _  / __ |/ /__/ (_-<    / /   / ___/ __/ _ \(_-</ __/ _ \/ -_) __/ / __/ / /__/ _ \/ _ \/ __/ __/ _ \/ / 
/_/ /_//_/_/ |_/____/_/___/   / /   /_/  /_/  \___/___/\__/_//_/\__/\__/_/\__/  \___/\___/_//_/\__/_/  \___/_/  
                             /_/                                                                                
EEG-Based Prosthetic Control System v.0.1.0                            
Written by Krishna Bhatt (krishbhatt2019@gmail.com)
Latest version: https://github.com/blizzard-labs/thalis-eeg-control                         
================================================================================================================
    '''
    print(header)
    
    parser = argparse.ArgumentParser(description="EEG-Based Prosthetic Control")
    characters = string.ascii_letters + string.digits
    
    help_msg = '''
        Thalamic Integration System: EEG-Based Prosthetic Control
        Usage: python main.py --mode <mode> --model <model_path> --name <run_name> [OPTIONS]
    '''
    
    parser.add_argument("--mode", type=str, choices=['default', 'collect', 'simulate', 'fine-tune', 'train'],
                        required=True, help="Operation mode: 'default', 'collect', or 'simulate'")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    
    parser.add_argument("--duration", type=int, default=100,
                        required=False, help="Duration for data collection in seconds (default: 100s)")
    
    args = parser.parse_args()
    print(f"Selected Mode: {args.mode}")
    
    
    if args.mode == 'default':
        '''
        DEFAULT MODE: Collect and process EEG data w/ pre-trained model and simulate prosthetic control.
        '''
        
        collect_data(background_prcs=True, graphing=True, duration=args.duration, run_name=args.name)
        
    if args.mode == 'collect':
        collect_data(background_prcs=False, graphing=True, duration=args.duration, run_name=args.name)