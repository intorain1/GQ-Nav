import json
import os

class BaseAgent(object):
    ''' Base class for an REVERIE agent to generate and save trajectories. '''

    def __init__(self, env):
        self.env = env
        self.results = {}

    def get_results(self):
        output = []
        for k, v in self.results.items():
            output.append({'instr_id': k, 'trajectory': v['path']})
        return output

    def _make_action(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        # self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self._make_action(**kwargs):
                    self.results[traj['instr_id']] = traj
                    preds_detail = self.get_results()
                    json.dump(
                    preds_detail,
                    open(os.path.join(self.config.log_dir, 'runtime.json'), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                    )
        else:   # Do a full round
            while True:
                for traj in self._make_action(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.results[traj['instr_id']] = traj
                        preds_detail = self.get_results()
                        json.dump(
                        preds_detail,
                        open(os.path.join(self.config.log_dir, 'runtime.json'), 'w'),
                        sort_keys=True, indent=4, separators=(',', ': ')
                        )
                if looped:
                    break
