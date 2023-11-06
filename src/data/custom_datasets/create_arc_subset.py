import json
from pathlib import Path

subset = [
    {'name': '00d62c1b.json', 'inputs': [0], 'outputs': []},
    {'name': '045e512c.json', 'inputs': [], 'outputs': [0, 1, 2]},
    {'name': '0962bcdd.json', 'inputs': [0, 1], 'outputs': []},
    {'name': '1caeab9d.json', 'inputs': [0, 1], 'outputs': []},
    {'name': '25d487eb.json', 'inputs': [0, 2], 'outputs': []},
    {'name': '31aa019c.json', 'inputs': [], 'outputs': [0]},
    {'name': '321b1fc6.json', 'inputs': [], 'outputs': [0]},
    {'name': '3aa6fb7a.json', 'inputs': [], 'outputs': [0]},
    {'name': '3befdf3e.json', 'inputs': [0, 1], 'outputs': []},
]

results = []

for entry in subset:
    fp = Path(f'../../../data/ARC-800-tasks/train/{entry["name"]}')
    with open(str(fp.absolute())) as f:
        data = json.load(f)


    for i in entry['inputs']:
        results.append({
            'data': data['train'][i]['input'],
            'metadata': {'task': entry['name'], 'index': i, 'type': 'input'}
        })

    for i in entry['outputs']:
        results.append({
            'data': data['train'][i]['output'],
            'metadata': {'task': entry['name'], 'index': i, 'type': 'output'}
        })

with open('../../../data/arc_subset.json', 'w') as f:
    json.dump(results, f, indent=4)



