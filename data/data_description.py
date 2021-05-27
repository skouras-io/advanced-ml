feature_names = [
  "ID",#
  "AGE",#
  "SEX",#
  "INF_ANAM",#
  "STENOK_AN",#
  "FK_STENOK",#
  "IBS_POST",#
  "IBS_NASL",#
  "GB",#
  "SIM_GIPERT",#
  "DLIT_AG",#
  "ZSN_A",#
  "nr11",#
  "nr01",#
  "nr02",#
  "nr03",#
  "nr04",#
  "nr07",#
  "nr08",#
  "np01",#
  "np04",#
  "np05",#
  "np07",#
  "np08",#
  "np09",#
  "np10",#
  "endocr_01",#
  "endocr_02",#
  "endocr_03",#
  "zab_leg_01",#
  "zab_leg_02",#
  "zab_leg_03",#
  "zab_leg_04",#
  "zab_leg_06",#
  "S_AD_KBRIG",#
  "D_AD_KBRIG",#
  "S_AD_ORIT",#
  "D_AD_ORIT",#
  "O_L_POST",#
  "K_SH_POST",#
  "MP_TP_POST",#
  "SVT_POST",#
  "GT_POST",#
  "FIB_G_POST",#
  "ant_im",#
  "lat_im",#
  "inf_im",#
  "post_im",#
  "IM_PG_P",#
  "ritm_ecg_p_01",#
  "ritm_ecg_p_02",#
  "ritm_ecg_p_04",#
  "ritm_ecg_p_06",#
  "ritm_ecg_p_07",#
  "ritm_ecg_p_08",#
  "n_r_ecg_p_01",#
  "n_r_ecg_p_02",#
  "n_r_ecg_p_03",#
  "n_r_ecg_p_04",#
  "n_r_ecg_p_05",#
  "n_r_ecg_p_06",#
  "n_r_ecg_p_08",#
  "n_r_ecg_p_09",#
  "n_r_ecg_p_10",#
  "n_p_ecg_p_01",#
  "n_p_ecg_p_03",#
  "n_p_ecg_p_04",#
  "n_p_ecg_p_05",#
  "n_p_ecg_p_06",#
  "n_p_ecg_p_07",#
  "n_p_ecg_p_08",#
  "n_p_ecg_p_09",#
  "n_p_ecg_p_10",#
  "n_p_ecg_p_11",#
  "n_p_ecg_p_12",#
  "fibr_ter_01",#
  "fibr_ter_02",#
  "fibr_ter_03",#
  "fibr_ter_05",#
  "fibr_ter_06",#
  "fibr_ter_07",#
  "fibr_ter_08",#
  "GIPO_K",#
  "K_BLOOD",#
  "GIPER_Na",#",#
  "Na_BLOOD",#
  "ALT_BLOOD",#
  "AST_BLOOD",#
  "KFK_BLOOD",#
  "L_BLOOD",#
  "ROE",#
  "TIME_B_S",#
  "R_AB_1_n",#
  "R_AB_2_n",#
  "R_AB_3_n",#
  "NA_KB",#
  "NOT_NA_KB",#
  "LID_KB",#
  "NITR_S",#
  "NA_R_1_n",#
  "NA_R_2_n",#
  "NA_R_3_n",#
  "NOT_NA_1_n",#
  "NOT_NA_2_n",#
  "NOT_NA_3_n",#
  "LID_S_n",#
  "B_BLOK_S_n",#
  "ANT_CA_S_n",#
  "GEPAR_S_n",#
  "ASP_S_n",#
  "TIKL_S_n",#
  "TRENT_S_n",#
  "FIBR_PREDS",#
  "PREDS_TAH",#
  "JELUD_TAH",#
  "FIBR_JELUD",#
  "A_V_BLOK",#
  "OTEK_LANC",#
  "RAZRIV",#
  "DRESSLER",#
  "ZSN",#
  "REC_IM",#
  "P_IM_STEN",#
  "LET_IS",#
]

feature_descriptions = [
  "1. Record ID",
  "2. Age",
  "3. Gender",
  "4. Quantity of myocardial infarctions in the anamnesis",
  "5. Exertional angina pectoris in the anamnesis",
  "6. Functional class",
  "7. Coronary heart disease",
  "8. Heredity on CHD",
  "9. Presence of an essential hypertension",
  "10. Symptomatic hypertension",
  "11. Duration of arterial hypertension",
  "12. Presence of chronic Heart failure",
  "13. Observing of arrhythmia in the anamnesis",
  "14. Premature atrial contractions in the anamnesis",
  "15. Premature ventricular contractions in the anamnesis",
  "16. Paroxysms of atrial fibrillation in the anamnesis",
  "17. A persistent form of atrial fibrillation in the anamnesis",
  "18. Ventricular fibrillation in the anamnesis",
  "19. Ventricular paroxysmal tachycardia in the anamnesis",
  "20. First-degree AV block in the anamnesis",
  "21. Third-degree AV block in the anamnesis",
  "22. LBBB",
  "23. Incomplete LBBB in the anamnesis",
  "24. Complete LBBB in the anamnesis",
  "25. Incomplete RBBB in the anamnesis",
  "26. Complete RBBB in the anamnesis",
  "27. Diabetes mellitus in the anamnesis",
  "28. Obesity in the anamnesis",
  "29. Thyrotoxicosis in the anamnesis",
  "30. Chronic bronchitis in the anamnesis",
  "31.Obstructive chronic bronchitis in the anamnesis",
  "32. Bronchial asthma in the anamnesis",
  "33. Chronic pneumonia in the anamnesis",
  "34. Pulmonary tuberculosis in the anamnesis",
  "35. Systolic blood pressure according to Emergency Cardiology Team",
  "36. Diastolic blood pressure according to Emergency Cardiology Team",
  "37. Systolic blood pressure according to intensive care unit",
  "38. Diastolic blood pressure according to intensive care unit",
  "39. Pulmonary edema at the time of admission to intensive care unit",
  "40. Cardiogenic shock at the time of admission to intensive care unit",
  "41. Paroxysms of atrial fibrillation at the time of admission to intensive care unit",
  "42. Paroxysms of supraventricular tachycardia at the time of admission to intensive care unit",
  "43. Paroxysms of ventricular tachycardia at the time of admission to intensive care unit",
  "44. Ventricular fibrillation at the time of admission to intensive care unit",
  "45. Presence of an anterior myocardial infarction",
  "46. Presence of a lateral myocardial infarction",
  "47. Presence of an inferior myocardial infarction",
  "48. Presence of a posterior myocardial infarction",
  "49. Presence of a right ventricular myocardial infarction",
  "50. ECG rhythm at the time of admission to hospital",
  "51. ECG rhythm at the time of admission to hospital",
  "52. ECG rhythm at the time of admission to hospital",
  "53. ECG rhythm at the time of admission to hospital",
  "54. ECG rhythm at the time of admission to hospital",
  "55. ECG rhythm at the time of admission to hospital",
  "56. Premature atrial contractions on ECG at the time of admission to hospital",
  "57. Frequent premature atrial contractions on ECG at the time of admission to hospital",
  "58.Premature ventricular contractions on ECG at the time of admission to hospital",
  "59. Frequent premature ventricular contractions on ECG at the time of admission to hospital",
  "60. Paroxysms of atrial fibrillation on ECG at the time of admission to hospital",
  "61. Persistent form of atrial fibrillation on ECG at the time of admission to hospital",
  "62. Paroxysms of supraventricular tachycardia on ECG at the time of admission to hospital",
  "63. Paroxysms of ventricular tachycardia on ECG at the time of admission to hospital",
  "64. Ventricular fibrillation on ECG at the time of admission to hospital",
  "65. Sinoatrial block on ECG at the time of admission to hospital",
  "66. First-degree AV block on ECG at the time of admission to hospital",
  "67. Type 1 Second-degree AV block",
  "68. Type 2 Second-degree AV block",
  "69. Third-degree AV block on ECG at the time of admission to hospital",
  "70. LBBB",
  "71. LBBB",
  "72. Incomplete LBBB on ECG at the time of admission to hospital",
  "73. Complete LBBB on ECG at the time of admission to hospital",
  "74. Incomplete RBBB on ECG at the time of admission to hospital",
  "75. Complete RBBB on ECG at the time of admission to hospital",
  "76. Fibrinolytic therapy by",
  "77. Fibrinolytic therapy by",
  "78. Fibrinolytic therapy by",
  "79. Fibrinolytic therapy by Streptase",
  "80. Fibrinolytic therapy by",
  "81. Fibrinolytic therapy by",
  "82. Fibrinolytic therapy by Streptodecase 1",
  "83. Hypokalemia",
  "84. Serum potassium content",
  "85. Increase of sodium in serum (more than 150 mmol/L)",
  "86. Serum sodium content",
  "87. Serum AlAT content",
  "88. Serum AsAT content",
  "89. Serum CPK content",
  "90. White blood cell count",
  "91. ESR",
  "92. Time elapsed from the beginning of the attack of CHD to the hospital",
  "93. Relapse of the pain in the first hours of the hospital period",
  "94. Relapse of the pain in the second day of the hospital period",
  "95. Relapse of the pain in the third day of the hospital period",
  "96. Use of opioid drugs by the Emergency Cardiology Team",
  "97. Use of NSAIDs by the Emergency Cardiology Team",
  "98.Use of lidocaine by the Emergency Cardiology Team",
  "99. Use of liquid nitrates in the ICU",
  "100. Use of opioid drugs in the ICU in the first hours of the hospital period",
  "101. Use of opioid drugs in the ICU in the second day of the hospital period",
  "102. Use of opioid drugs in the ICU in the third day of the hospital period",
  "103. Use of NSAIDs in the ICU in the first hours of the hospital period",
  "104. Use of NSAIDs in the ICU in the second day of the hospital period",
  "105. Use of NSAIDs in the ICU in the third day of the hospital period",
  "106. Use of lidocaine in the ICU",
  "107. Use of beta-blockers in the ICU",
  "108. Use of calcium channel blockers in the ICU",
  "109. Use of",
  "110. Use of acetylsalicylic acid in the ICU",
  "111. Use of Ticlid in the ICU",
  "112. Use of Trental in the ICU",
  "113. Atrial fibrillation",
  "114. Supraventricular tachycardia",
  "115. Ventricular tachycardia",
  "116. Ventricular fibrillation",
  "117. Third-degree AV block",
  "118. Pulmonary edema",
  "119. Myocardial rupture",
  "120. Dressler syndrome",
  "121. Chronic heart failure",
  "122. Relapse of the myocardial infarction",
  "123. Post-infarction angina",
  "124. Lethal outcome",
]