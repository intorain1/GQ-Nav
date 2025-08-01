import os
import json
import time

from env_src.utils.data_utils import construct_instrs
from env_src.utils.logger import write_to_record_file

from env_src.utils.data import ImageObservationsDB
from parser import parse_args
from env import R2RNavBatch
from agent import NavAgent
import argparse
import tqdm
from graph import OptimizedTimeObjectGraph

def get_results(results):
    output = []
    for k, v in results.items():
        output.append({'instr_id': k, 'trajectory': v['path']})
    return output

def build_dataset(args):

    feat_db = ImageObservationsDB(args.obj_dir)

    dataset_class = R2RNavBatch

    val_env_names = [args.val_env_name]

    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split]
        )
        
        val_env = dataset_class(
            feat_db, val_instr_data, args.connectivity_dir, args.navigable_dir,
            batch_size=args.batch_size, seed=args.seed, name=split,
        )   # evaluation using all objects
        val_envs[split] = val_env

    return val_envs

def valid(args, val_envs):
    results = []
    env = next(iter(val_envs.values()))
    num_episodes = len(env.data)
    # print(num_episodes)

    for i in tqdm.tqdm(range(num_episodes), desc='testing'):
        env = next(iter(val_envs.values()))
        agent = NavAgent(env, args)
        traj = agent.get_obj_traj()
        results.append(traj)

    return results

if __name__ == "__main__":
    args = parse_args()
    val_envs = build_dataset(args)

    results = valid(args, val_envs)
    print(len(results))
    for i in tqdm.tqdm(range(len(results)),desc='processing'):
        title = results[i][0]['instr_id']
        obj = results[i][0]['obj']
        graph = OptimizedTimeObjectGraph()
        for j in range(len(obj)):
            graph.add_recognition(j, obj[j])
        graph.visualize('/home/mspx/icra/GQ-Nav/data_traj', title)
        # print(results[i]['instr_id'], results[i]['obj'])
    # output_dir = '/home/mspx/icra/GQ-Nav/results.json'
    # with open(output_dir, 'w') as f:
    #     json.dump(results, f, indent=2)
    
    
