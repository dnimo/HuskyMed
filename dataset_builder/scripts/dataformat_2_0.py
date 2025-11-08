import itertools
import os
from datetime import datetime

import pandas as pd
from datasets import load_dataset


def main():
    data_path = "/home/jovyan/zhang/Husky/Unit/2022/sample/infotree_mrsmr_000_20240116_114018_149346.json"
    data = load_dataset("json", data_files=data_path)

    df = pd.DataFrame(data["train"])

    reform = []

    for _, items in df.iterrows():
        result = []

        for item in items["smrs"]:
            if item["smr"] is not None:
                item["INPUT_DATE"] = datetime.strptime(item["INPUT_DATE"], "%Y-%m-%d")
        for item in items["records"]:
            item["ts"] = datetime.strptime(item["ts"][:8], "%Y%m%d")

        smrs = items["smrs"]
        records = items["records"]

        for smr_item in smrs:
            insert_index = next(
                (
                    i
                    for i, record_item in enumerate(records)
                    if record_item["ts"] > smr_item["INPUT_DATE"]
                ),
                len(smrs),
            )

            result.extend(records[:insert_index])
            result.append(smr_item)
            records = records[insert_index:]

        result.extend(records)
        reform.append({items["pt_id"]: result})

    df_reform = pd.DataFrame(reform)

    print("Unit is loaded")

    def merge_items(data):
        for i, element in enumerate(data):
            if "smr" in element:
                df_ref_list = data[:i]
                break
        else:
            return []

        df_ref = pd.DataFrame(df_ref_list)
        df_target = pd.DataFrame(element["smr"])

        merged_list = []
        if not df_ref.columns.empty and not df_target.columns.empty:
            df_ref_grouped = df_ref.groupby("data_type")
            df_target_grouped = df_target.groupby("data_type")
            for data_type, group in df_ref_grouped:
                if data_type in df_target_grouped.groups:
                    list1 = list(set(group["text"].to_list()))
                    list2 = list(set(df_target_grouped.get_group(data_type)["text"].to_list()))
                    merged_list.extend([(data_type,) + item for item in itertools.product(list1, list2)])
                elif data_type in ["S", "O", "A", "P"] and "治療経過" in df_target_grouped.groups:
                    list1 = list(set(group["text"].to_list()))
                    list2 = list(set(df_target_grouped.get_group("治療経過")["text"].to_list()))
                    merged_list.extend([("治療経過",) + item for item in itertools.product(list1, list2)])

        return merged_list

    df_reform["merged"] = df_reform.apply(lambda row: merge_items(row[next(iter(row.keys()))]), axis=1)

    print("Unit is merged")

    df_reform = df_reform[["section", "smr_type", "merged"]]

    print("selected columns")

    group_size = 500
    dfs = [df_reform.iloc[i : i + group_size] for i in range(0, len(df_reform), group_size)]
    output_folder = "/home/jovyan/public/zhang/train/tfidf/"
    os.makedirs(output_folder, exist_ok=True)
    for i, chunk_df in enumerate(dfs):
        filename = os.path.join(output_folder, f"instrucation_data_chunk_{i + 1}.json")
        chunk_df.to_json(filename, orient="records")

    print("Unit is saved")


if __name__ == "__main__":
    main()
