import os

# os.system(
#     "python scripts/compute_metrics.py --sub --out trained_mcse_chime3/pred_deftan_vtest_100 --src ~/datasets/CHiME3/data/audio/16kHz/isolated/test --metrics asr pesqn pesqw stoi --excel DeFT-AN.xlsx"
# )

os.system(
    "python scripts/compute_metrics.py --sub --out trained_mcse_chime3/pred_mcse_3en_wdefu_wfafu_c72_B2_test_100 --src ~/datasets/CHiME3/data/audio/16kHz/isolated/test --metrics asr pesqn pesqw stoi --excel FuNet.xlsx"
)

os.system(
    "python scripts/compute_metrics.py --sub --out trained_mcse_chime3/pred_mcnet_vtest_100 --src ~/datasets/CHiME3/data/audio/16kHz/isolated/test --metrics asr pesqn pesqw stoi --excel McNet.xlsx"
)
