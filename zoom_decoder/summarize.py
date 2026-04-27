import json, glob, os
results = {}
for p in sorted(glob.glob('./zoom_decoder/eval/*.json')):
    name = os.path.basename(p).replace('.json', '')
    d = json.load(open(p))['summary']
    results[name] = d

header = [4, 8, 12, 16, 24, 32, 48]
print(f'{"config":<16} {"overall":>8}  ' + '  '.join(f'px={h:<4}' for h in header))
for name in ['base', 'zd_6L', 'zd_12L', 'zd_24L', 'zd_24L_noap', 'zd_24L_nosw', 'lora_r16']:
    if name not in results:
        continue
    r = results[name]
    by = r['by_pixel_size']
    row = [f'{r["overall_acc"]:>7.2f}%']
    for ps in header:
        v = by.get(str(ps))
        row.append(f'{v["acc"]:>5.1f}%' if v else '  --  ')
    print(f'{name:<16} ' + '  '.join(row))

print()
print('=== by task_type ===')
tasks = sorted(results['base']['by_task'].keys())
print(f'{"config":<16} ' + '  '.join(f'{t[:10]:>10}' for t in tasks))
for name in ['base', 'zd_24L', 'zd_24L_nosw', 'lora_r16']:
    if name not in results:
        continue
    r = results[name]['by_task']
    print(f'{name:<16} ' + '  '.join(f'{r[t]["acc"]:>9.1f}%' for t in tasks))
