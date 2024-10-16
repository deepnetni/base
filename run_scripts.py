import os

# fmt: off
os.system("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_3en_wdefu_wfafu_cau_c72_B2")
os.system("python train_spatial_chime.py")
os.system("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_baseline+TC+ED")

os.system("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_baseline")
os.system("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_uc_wdefu_wfafu_c72_B2")
os.system("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_baseline+UC+ED")
os.system("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_baseline+3en+ED")
os.system("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_3en_all_aff_wdefu_wfafu_c72_B2")
os.system("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_3en_all_att_wdefu_wfafu_c72_B2")
# os.system("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_baseline+3en+ED+FA")
os.system("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_baseline+3en")
os.system("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_baseline+3aff")
os.system("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_baseline+3att")

# os.system("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_3en_wdefu_wfafu_c72_B2")
os.system("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_3en_wdefu_wfafu_inv_c72_B2")
