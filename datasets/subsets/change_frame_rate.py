"""change sdd frame rate of 1 to trajnet_sdd frame rate of 12"""
import os

def save_new_cfg(filename, cfg):
    print("cfg:", cfg)
    print(f"save to {filename}?")
    # import ipdb; ipdb.set_trace()
    with open(filename, 'w') as f:
        f.write(cfg)


def main():
    total_new_cfgs = 0
    for file in os.listdir('.'):
        if not file.endswith('.plist'):
            continue
        with open(file, 'r') as f:
            cfg = f.read()
        cfg2 = cfg.replace('''<key>paras</key>
	<array>
		<integer>1</integer>''', f'''<key>paras</key>
	<array>
		<integer>12</integer>''')
        save_new_cfg(file, cfg2)
        total_new_cfgs += 1


if __name__ == '__main__':
    main()