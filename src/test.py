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
        traj = agent._make_action(threshold=0.5)
        results.append(traj)

    for env_name, env in val_envs.items():
        score_summary, _ = env.eval_metrics(results)
        loss_str = "Env name: %s" % env_name
        for metric, val in score_summary.items():
            loss_str += ', %s: %.2f' % (metric, val)

    return loss_str

if __name__ == "__main__":
    args = parse_args()
    val_envs = build_dataset(args)

    results = valid(args, val_envs)
    print(results)
    # results = []


    # # Get the number of episodes in the first validation environment
    # env = next(iter(val_envs.values()))
    # num_episodes = len(env.data)  # Assuming 'data' holds the episodes

    # agent = NavAgent(env, args)
    # traj = agent._make_action(threshold=0.5)  # Example threshold
    # results.append(traj)

    # agent = NavAgent(next(iter(val_envs.values())), args)
    # traj = agent._make_action(threshold=0.5)
    # results.append(traj)

    # preds = results

    # print(preds)

    # for env_name, env in val_envs.items():
    #     score_summary, _ = env.eval_metrics(preds)
    #     loss_str = "Env name: %s" % env_name
    #     for metric, val in score_summary.items():
    #         loss_str += ', %s: %.2f' % (metric, val)
    #     print(loss_str)

    # print('-----------------------------------')
    # start_time = time.time()
    # for i in range(0,5):
    #     if i == 0:
    #         agent._make_action(threshold=0.5)  # Example threshold
    #     else:
    #         agent._make_action(threshold=0.5, reset=False)
    # end_time = time.time()
    # print(f"Total time taken for 5 actions: {end_time - start_time:.2f} seconds")
    # agent.rollout(reset=True)
    # agent.rollout(reset=False)
    
