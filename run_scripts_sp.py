import os

# fmt: off
os.system("python train_mcse_spdns.py --conf config/yconf_mcse_sp.yaml --name mcse_baseline+3en")
os.system("python train_mcse_spdns.py --conf config/yconf_mcse_sp.yaml --name mcse_baseline")
os.system("python train_mcse_spdns.py --conf config/yconf_mcse_sp.yaml --name mcse_baseline+3en+ED")
# os.system("python train_mcse_spdns.py --conf config/yconf_mcse_sp.yaml --name mcse_baseline+3en+ED+FA")
os.system("python train_mcse_spdns.py --conf config/yconf_mcse_sp.yaml --name mcse_baseline+3aff")
os.system("python train_mcse_spdns.py --conf config/yconf_mcse_sp.yaml --name mcse_baseline+3att")

os.system("python train_mcnet_spdns.py --conf config/yconf_mcnet_sp.yaml")
os.system("python train_deftan_spdns.py --conf config/yconf_deftan_sp.yaml")
