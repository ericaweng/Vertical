"""parallelize evaluate all"""
import os
import argparse
import subprocess
import torch
import json
import multiprocessing
from pathlib import Path
from itertools import product


def get_cmds(args):
    cmds = []
    if args.mode == 'train':
        for weights in ['0.6,0.4', '0.5,0.5', '0.7,0.3', '0.8,0.2']:
            for dataset in 'eth hotel univ zara1 zara2 sdd'.split():
                cmd = f'python main.py --model va --key_points 3_7_11 --test_set {dataset} --keypoints_loss_type mix --loss_weights_a {weights} --metric sfde'
                cmds.append(cmd)
        for weights in ['0.5,0.3,0.2', '0.4,0.4,0.2', '0.6,0.2,0.2']:
            for dataset in 'eth hotel univ zara1 zara2 sdd'.split():
                cmd = f'python main.py --model vb --points 3 --test_set {dataset} --loss_type mix --loss_weights_b {weights} --metric sfde'
                cmds.append(cmd)
    if args.mode == 'ade':
        return [
            "python main.py   --model V --loada ./logs/20230225-190502modelvaeth   --loadb  ./logs/20230225-193323modelvbeth",
            "python main.py   --model V --loada ./logs/20230225-190739modelvahotel   --loadb ./logs/20230225-193331modelvbhotel",
            "python main.py   --model V --loada ./logs/20230225-193037modelvauniv  --loadb ./logs/20230225-192959modelvbuniv",
            "python main.py   --model V --loada ./logs/20230225-194144modelvazara1   --loadb   ./logs/20230225-194159modelvbzara1",
            "python main.py   --model V --loada ./logs/20230225-194045modelvazara2   --loadb ./logs/20230225-204854modelvbzara2",
            "python main.py   --model V --loada ./logs/20230227-123855modelvasdd  --loadb  ./logs/20230227-123842modelvbsdd" ]
    if args.mode == 'sade':
        return """ipy main.py   --model V --loada ./logs/20230227-220653modelvaeth --loadb ./logs/20230227-230244modelvbeth --save_traj_dir vv_jade
        ipy main.py   --model V --loada ./logs/20230227-230701modelvahotel --loadb ./logs/20230227-230701modelvbhotel --save_traj_dir vv_jade
        ipy main.py   --model V --loada ./logs/20230227-230701modelvauniv --loadb ./logs/20230227-230701modelvbuniv --save_traj_dir vv_jade
        ipy main.py   --model V --loada ./logs/20230227-230701modelvazara1 --loadb ./logs/20230227-230701modelvbzara1 --save_traj_dir vv_jade
        ipy main.py   --model V --loada ./logs/20230227-230701modelvazara2 --loadb ./logs/20230227-232350modelvbzara2 --save_traj_dir vv_jade
        ipy main.py   --model V --loada ./logs/20230227-233707modelvasdd --loadb ./logs/20230227-233707modelvbsdd --save_traj_dir vv_jade""".replace(
                'ipy', 'python').splitlines()

    if args.mode == 'mix_loss_old':
        vas = {}
        vbs = {}
        for model_path in Path('logs').glob('20230314-*'):
            model_path = str(model_path)
            if 'modelva' in model_path:
                loss_weights = "vv_lw_" + "-".join([f"{lw:0.1f}" for lw in json.load(open(f'{model_path}/args.json'))['loss_weights_a']])
                vas[loss_weights] = model_path
                print(f"loss_weights a: {loss_weights}")
            if 'modelvb' in model_path:
                loss_weights = "vv_lw_" + "-".join([f"{lw:0.1f}" for lw in json.load(open(f'{model_path}/args.json'))['loss_weights_b']])
                vbs[loss_weights] = model_path
                print(f"loss_weights b: {loss_weights}")

        import ipdb; ipdb.set_trace()
        assert len(vas) == len(vbs), f"{len(vas)} {len(vbs)}"
        assert "--".join(sorted(list(va.keys()))) == "--".join(sorted(list(vb.keys()))), f"{va.keys()} {vb.keys()}"
        for save_dir, va in vas.items():
            vb = vbs[save_dir]
            cmd = f"python main.py   --model V --loada {va} --loadb {vb} --save_traj_dir {save_dir}"
            cmds.append(cmd)

    if args.mode == 'mix_loss':
        for dataset in ['eth', 'hotel', 'univ', 'zara1', 'zara2', 'sdd']:
            vas = {}
            vbs = {}
            for model_path in Path('logs').glob(f'20230527*{dataset}*'):
                # print(f"dataset: {dataset}")
                model_path = str(model_path)
                if 'modelva' in model_path:
                    loss_weights = ",".join([f"{lw:0.1f}" for lw in json.load(open(f'{model_path}/args.json'))['loss_weights_a']])
                    # print(f"loss_weights a: {loss_weights}")
                    # if loss_weights != '0.2,0.8':
                    #     continue
                    vas[loss_weights] = model_path
                if 'modelvb' in model_path:
                    loss_weights = ''.join(json.load(open(f'{model_path}/args.json'))['loss_weights_b'])
                    # print(f"loss_weights b: {loss_weights}")
                    # if loss_weights != '0.3,0.5,0.2':
                    #     continue
                    vbs[loss_weights] = model_path

            for (weights_a, va), (weights_b, vb) in product(vas.items(), vbs.items()):
                save_dir = "vv_ml_a-" + weights_a + "_b-" + weights_b
                cmd = f"python main.py   --model V --loada {va} --loadb {vb} --save_traj_dir {save_dir}"
                print(f"cmd: {cmd}")
                cmds.append(cmd)
    exit()

    return cmds


def spawn(cmds, args):
    """launch cmds in separate threads, max_cmds_at_a_time at a time, until no more cmds to launch"""
    print(f"launching at most {args.max_cmds_at_a_time} cmds at a time:")

    sps = []
    num_gpus = len(args.gpus_available)
    total_cmds_launched = 0  # total cmds launched so far
    cmds = cmds[args.start_from:]

    while total_cmds_launched < len(cmds):
        cmd = cmds[total_cmds_launched]
        # assign gpu and launch on separate thread
        gpu_i = args.gpus_available[total_cmds_launched % num_gpus]
        # env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu_i)}
        cmd = cmd + f' --gpu {gpu_i}'
        print(gpu_i, cmd)
        if not args.trial:
            if args.redirect_output:
                output_filename = 'logs_output'
                # output_filename = cmd.replace(' ', '_').replace('/', '_').replace('=', '_').replace('-', '_').replace('.', '_')
                cmd = f"sudo {cmd} >> {output_filename}.txt 2>&1"
            # cmd = f"sudo {cmd}"
            sp = subprocess.Popen(cmd, env=os.environ, shell=True)
            sps.append(sp)
            if len(sps) >= args.max_cmds_at_a_time:
                # this should work if all subprocesses take the same amount of time;
                # otherwise we might be waiting longer than necessary
                sps[0].wait()
                sps = sps[1:]
        total_cmds_launched += 1

    print("total cmds launched:", total_cmds_launched)
    [sp.wait() for sp in sps]
    print(f"finished all {total_cmds_launched} processes")


def main(args):
    cmds = get_cmds(args)
    spawn(cmds, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_cmds', '-mc', type=int, default=100)
    parser.add_argument('--max_cmds_at_a_time', '-c', type=int, default=max(1, multiprocessing.cpu_count() - 3))
    parser.add_argument('--start_from', '-sf', type=int, default=0)
    try:
        cuda_visible = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    except KeyError:
        cuda_visible = list(range(torch.cuda.device_count()))
    parser.add_argument('--gpus_available', '-ga', nargs='+', type=int, default=cuda_visible)
    parser.add_argument('--no_trial', '-nt', dest='trial', action='store_false',
                        help='if not trial, then actually run the commands')
    parser.add_argument('--redirect_output', '-ro', action='store_true')
    parser.add_argument('--datasets', '-d', nargs='+', type=str, default=None)
    parser.add_argument('--aggregations', '-a', nargs='+', type=str, default=['min', 'mean'])
    parser.add_argument('--metrics', '-mr', nargs='+', type=str,
                        default=['ade', 'fde', 'joint_ade', 'joint_fde', 'col_pred'])
    parser.add_argument('--num_samples', '-ns', nargs='+', type=int, default=[20])
    parser.add_argument('--skip_existing', '-se', action='store_true')
    parser.add_argument('--glob_str', nargs='+', default=None)
    parser.add_argument('--mode', '-m', default='train')

    args = parser.parse_args()
    main(args)
