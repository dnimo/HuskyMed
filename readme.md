## format funcation

this new strucature is design for instrucation tuning

```python
# def chunk_examples_parallel(items):
#     result = []
#     refs = pd.
#     targets = {item['data_type']: item['text'] for item in items}
#     prompt = 'instruction:' + 'あなたは'+ items['section'] +'の医師です。以下は時系列に並べられた患者の臨床記録です。以下の記録に基づいて'+ items['smr_type'] +'を作成してください。'+ '\n'
#     # for ref in refs:
#     #     for tar in targets:
#     #         if ref['data_type'] == tar['data_type']:
#     #             sample = prompt + "記録カテゴリ：" + ref['data_type'] + ' ' + '記録内容：' + ref['text'] + '\n' + 'Response:' + "記録カテゴリ：" + tar['data_type'] + ' ' + '記録内容：' + tar['text'] + '\n'
#     #             result.append(sample)
#     #         if ref['data_type'] in ['S', 'O', 'A', 'P'] and tar['data_type'] == '治療経過':
#     #             sample = prompt + "記録カテゴリ：" + ref['data_type'] + ' ' + '記録内容：' + ref['text'] + '\n' + 'Response:' + "記録カテゴリ：" + tar['data_type'] + ' ' + '記録内容：' + tar['text'] + '\n'
#     #             result.append(sample)
#     for item in refs
#     return result

```
the old one

```python
# def chunk_examples(examples):
#     result = []
#     for items in examples:
#         chunks = []
#         ref = items['ref']
#         smr = "\n".join(items['smr_text'])
#         for index, item in enumerate(ref):
#             chunks.append("## 治療記録"+ '\n' + "#### " + item['data_type'] + '\n' + item['text'] + '\n')
            
#         result.append({"ref": chunks, 'smr_type':items['smr_type'], "smr": smr})
#     return {'Unit': result}
```