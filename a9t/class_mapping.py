# NOTE: The labels are as per harmonization scheme
# WARNING: Label Ordering VVV important, overwriting possible. Put largest labels first
"""
Anorectum
Bladder
Bag_Bowel
Femur_L
Femur_R
Penilebulb
Femur_Head_L
"""
ALL_SEG_MAP = {

    "TSPrimeOrgans": {
        # Harmonized
        1: "Bladder",
        2: "Anorectum",
        3: "Bag_Bowel",
        4: "Femur_Head_L",
        5: "Femur_Head_R",
        6: "Penilebulb",
    },
    "TSPrimeCTVP": {
        1: "Ctvp"
    },
    "TSPrimeCTVN": {
        1: "Ctvn"
    },
    "TSGyne": {
        1: "Body",
        2: "Bladder",
        3: "Bag_Bowel",
        4: "SpinalCord",
        5: "Duodenum",
        6: "Pancreas",
        7: "Rectum",
        8: "Colon_Sigmoid",
        9: "Vs_Int_Iliac",
        10: "Vs_Common_Iliac",
        11: "Vs_Ext_Iliac",
        12: "Vagina",
        13: "V_Venacava_I",
        14: "Stomach",
        15: "Spleen",
        16: "Liver",
        17: "Kidney_R",
        18: "Kidney_L",
        19: "Femur_R",
        20: "Femur_L",
        21: "Ctvn_Presacral",
        22: "Ctvn_Paraaortic",
        23: "Ctvn_Obturator",
        24: "Ctvn_Int_Iliac",
        25: "Ctvn_Ext_Iliac",
        26: "Ctvn_Com_Iliac",
        27: "Bowel",
        28: "Bone_Marrow_Pel",
        29: "Bone_Marrow_Lum",
        30: "Anus",
        31: "A_Aorta",
    },
}
