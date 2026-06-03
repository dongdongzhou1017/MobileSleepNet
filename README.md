# MobileSleepNet
Codes and Video demonstration for the paper titled "MobileSleepNet: A Deployment-Oriented Framework for Real-Time Sleep Staging on Mobile Devices".

This paper has been submitted to IEEE Journal of Biomedical and Health Informatics (IEEE JBHI). 

## Quick Start
### 1. Prepare Data
Download the Sleep-EDF 2013 dataset and prepare it in .npz format. Update the data path in the configuration:
DATA_PATH = "./datasets/sleepedf-2013/npz/sleep_edf_processed_Fpz.npz"
### 2. Training model
python MobileSleepNet.py
