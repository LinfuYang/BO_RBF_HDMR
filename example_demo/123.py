import pandas as pd

list_1 = [0, 1, 2, 3]
df_1 = pd.DataFrame(data=list_1)
df_1.to_csv('../results_jump/GMF_jump_1.csv', sep='\t')