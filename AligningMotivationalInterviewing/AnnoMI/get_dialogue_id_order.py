"""
The script runs once to determine the order of dialogue to be used randomly.
Note that train/test split is in the same order as the public dataset though
"""
# import pandas as pd
# annomi_train = pd.read_csv("""train.csv""")
# high_quality_mi_dialogue_ids = sorted(list(set(annomi_train[annomi_train['mi_quality'] == 'high']['transcript_id'].unique())))
# low_quality_mi_dialogue_ids = sorted(list(set(annomi_train[annomi_train['mi_quality'] == 'low']['transcript_id'].unique())))
# import random 
# random.seed(0)
# random.shuffle(high_quality_mi_dialogue_ids)
# random.shuffle(low_quality_mi_dialogue_ids)
# pd.DataFrame({'id':high_quality_mi_dialogue_ids}).to_csv('random_order_high_quality_dialogue_ids.csv')
# pd.DataFrame({'id':low_quality_mi_dialogue_ids}).to_csv('random_order_low_quality_dialogue_ids.csv')