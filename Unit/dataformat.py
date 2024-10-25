import itertools
from datetime import datetime

import pandas as pd
from datasets import load_dataset

data_path = '/home/jovyan/zhang/Husky/Unit/2022/pt_id_split_mrsmr'
data = load_dataset('json', data_dir=data_path, num_proc=8)

df = pd.DataFrame(data["train"])

reform = []

for index, items in df.iterrows():
    smrs_list = []
    refs_list = []

    for item in items['smrs']:
        if item['smr'] != None:
            smrs_list.append(
                {'INPUT_DATE': datetime.strptime(item['INPUT_DATE'], '%Y-%m-%d'), 'smr_type': item['EMR_TYPE'],
                 'section': item['SECTION'],
                 'Unit': [{'data_type': value['EMR_DATA_TYPE'], 'text': value['EMR_TEXT']} for value in
                              item['smr']]})
    for d in items['records']:
        ts = datetime.strptime(d['ts'][:8], '%Y%m%d')
        refs_list.append({'INPUT_DATE': ts, 'section': d['section'], 'Unit': d['emr']})

    for smr in smrs_list:
        new_refs = []
        for ref in refs_list:
            if ref['INPUT_DATE'] <= smr['INPUT_DATE']:
                new_refs.extend(ref['Unit'])
        reform.append(
            {'ref': new_refs, 'section': smr['section'], 'smr_type': smr['smr_type'], 'target': smr['Unit']})

df_reform = pd.DataFrame(reform)

print("Unit is loaded")


def merge_items(ref_list, target_list):
    df_ref = pd.DataFrame(ref_list)
    df_target = pd.DataFrame(target_list)

    merged_list = []
    if not df_ref.columns.empty and not df_target.columns.empty:
        df_ref_grouped = df_ref.groupby('data_type')
        df_target_grouped = df_target.groupby('data_type')
        for data_type, group in df_ref_grouped:
            if data_type in df_target_grouped.groups:
                list1 = list(set(group['text'].to_list()))
                list2 = list(set(df_target_grouped.get_group(data_type)['text'].to_list()))
                merged_list.extend([(data_type,) + item for item in itertools.product(list1, list2)])
            elif data_type in ['S', 'O', 'A', 'P'] and '治療経過' in df_target_grouped.groups:
                list1 = list(set(group['text'].to_list()))
                list2 = list(set(df_target_grouped.get_group('治療経過')['text'].to_list()))
                merged_list.extend([('治療経過',) + item for item in itertools.product(list1, list2)])

        return merged_list


# 对DataFrame应用合并字典的函数
df_reform['merged'] = df_reform.apply(lambda row: merge_items(row['ref'], row['target']), axis=1)

print("Unit is merged")

df_reform = df_reform[['section', 'smr_type', 'merged']]

print("selected columns")

import os

# 每500个元素为一组
group_size = 500

# 切分大的DataFrame为小的DataFrame列表
dfs = [df_reform.iloc[i:i + group_size] for i in range(0, len(df_reform), group_size)]
output_folder = '/home/jovyan/public/zhang/train/instrucation/'
os.makedirs(output_folder, exist_ok=True)
for i, chunk_df in enumerate(dfs):
    filename = os.path.join(output_folder, f'instrucation_data_chunk_{i + 1}.json')
    chunk_df.to_json(filename, orient='records')

print("Unit is saved")
