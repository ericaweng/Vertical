"""parallelize evaluate all"""
import os
import argparse
import subprocess
import torch
import multiprocessing
from pathlib import Path

def get_cmds(args):
    cmds = []
    for dataset in 'eth hotel'.split(): # sdd
        cmd = f'python main.py --model va --key_points 3_7_11 --test_set {dataset} --gpu 2'# --use_maps 0'
        # cmds.append(cmd)
        # cmd = f'python main.py --model vb --points 3 --test_set {dataset} --gpu 2'# --use_maps 0'
        cmds.append(cmd)
        # cmd = f'python main.py  --model V  --loada ./weights/vertical/a_{dataset} --loadb ./weights/vertical/b_{dataset}'
    cmds = [
            "python main.py   --model V --loada ./logs/20230225-190502modelvaeth   --loadb  ./logs/20230225-193323modelvbeth",
            "python main.py   --model V --loada ./logs/20230225-190739modelvahotel   --loadb ./logs/20230225-193331modelvbhotel",
            "python main.py   --model V --loada ./logs/20230225-193037modelvauniv  --loadb ./logs/20230225-192959modelvbuniv",
            "python main.py   --model V --loada ./logs/20230225-194144modelvazara1   --loadb   ./logs/20230225-194159modelvbzara1",
            "python main.py   --model V --loada ./logs/20230225-194045modelvazara2   --loadb ./logs/20230225-204854modelvbzara2", ]
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
        print(gpu_i, cmd)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu_i)}
        if not args.trial:
            if args.redirect_output:
                output_filename = 'logs_output'
                # output_filename = cmd.replace(' ', '_').replace('/', '_').replace('=', '_').replace('-', '_').replace('.', '_')
                cmd = f"sudo {cmd} >> {output_filename}.txt 2>&1"
            # cmd = f"sudo {cmd}"
            sp = subprocess.Popen(cmd, env=env, shell=True)
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
    parser.add_argument('--methods', '-m', nargs='+', type=str, default=None)
    parser.add_argument('--datasets', '-d', nargs='+', type=str, default=None)
    parser.add_argument('--aggregations', '-a', nargs='+', type=str, default=['min', 'mean'])
    parser.add_argument('--metrics', '-mr', nargs='+', type=str,
                        default=['ade', 'fde', 'joint_ade', 'joint_fde', 'col_pred'])
    parser.add_argument('--num_samples', '-ns', nargs='+', type=int, default=[20])
    parser.add_argument('--skip_existing', '-se', action='store_true')
    parser.add_argument('--glob_str', nargs='+', default=None)

    args = parser.parse_args()
    main(args)
