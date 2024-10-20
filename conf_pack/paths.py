import os
DATA_PARENT_PATH = 'C:/Users/orsym/Documents/Data'

# ------------------------ Stroke -----------------------------------------
STROKE_DATA_PATH = os.path.join(DATA_PARENT_PATH, 'Stroke')
SCANS_DIR_BEFORE = os.path.join(STROKE_DATA_PATH, 'Before')
SCANS_DIR_AFTER = os.path.join(STROKE_DATA_PATH, 'After')
STROKE_EXCEL_DATA = os.path.join(STROKE_DATA_PATH, 'Clinical_Abilities.xlsx')
STROKE_SAVE_PATH_PARENT = os.path.join(STROKE_DATA_PATH, 'Results')
# -----------------------ADHD ------------------------------------------
ADHD_DATA_PATH = os.path.join(DATA_PARENT_PATH, 'ADHD')
ADHD_EXCEL_DATA = os.path.join(ADHD_DATA_PATH, 'Data.xlsx')
ADHD_SAVE_PATH_PARENT = os.path.join(ADHD_DATA_PATH, 'Results')
