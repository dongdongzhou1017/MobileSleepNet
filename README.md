# MobileSleepNet
Codes and Video demonstration for the paper titled "MobileSleepNet: A Deployment-Oriented Framework for Real-Time Sleep Staging on Mobile Devices".

This paper has been submitted to IEEE Journal of Biomedical and Health Informatics(IEEE JBHI). 

Data Preparation：
Download the Sleep-EDF 2013 dataset and prepare it in .npzformat. Update the data path in the configuration:
DATA_PATH = "./datasets/sleepedf-2013/npz/sleep_edf_processed_Fpz.npz"

Training：
python MobileSleepNet.py
