import os
import logging
from pypmj import (jcm, _config, ResourceManager, SimulationSet)

class Optimizer(object):
    def __init__(self, project, domain, constraints=[], constant_keys={}, max_iter=20, num_parallel=0):
        self.logger = logging.getLogger('core.' + self.__class__.__name__)
        self.__project = project
        self.domain = domain
        self.constraints = constraints
        self.constant_keys = constant_keys
        self.max_iter = max_iter
        self.__num_parallel = num_parallel if num_parallel > 0 else ResourceManager().get_current_resources().get_resources()[0].multiplicity
        
        self.__domain_keys = []
        for i in range(len(self.domain)):
            self.__domain_keys.append(self.domain[i]['name'])
        
        self.study = jcm.optimizer.create_study(domain=self.domain, constraints=self.constraints)
        self.study.set_parameters(max_iter=self.max_iter, num_parallel=self.__num_parallel)
        
    def run(self, objective_func, storage_folder='from_date', storage_base='from_config', processing_func=None, transitional_storage_base=None, auto_rerun_failed=1, run_post_process_files=None):
        while (not self.study.is_done()):
            suggestions = []
            suggestion_ids = []
            for i in range(self.__num_parallel):
                suggestions.append(self.study.get_suggestion())
                suggestion_ids.append(suggestions[i].id)
                if self.study.info()['is_done']:
                    break
                
            parameter_keys = {'suggestion_id': suggestion_ids}
            geometry_keys = dict()
            for key in self.__domain_keys:
                values = []
                for suggestion in suggestions:
                    values.append(suggestion.kwargs[key])
                geometry_keys[key] = values
                
            template_keys = {
                'constants': self.constant_keys,
                'parameters': parameter_keys,
                'geometry': geometry_keys
            }
            
            simuset = SimulationSet(self.__project, template_keys, combination_mode='list', storage_folder=storage_folder, storage_base=storage_base, transitional_storage_base=transitional_storage_base)
            simuset.make_simulation_schedule()
            simuset.run(processing_func=processing_func, auto_rerun_failed=auto_rerun_failed, run_post_process_files=run_post_process_files, wdir_mode='delete')
            simuset.close_store()
            self.__clear_storage_dir(simuset)
            
            for i in range(len(simuset.simulations)):
                sid = simuset.simulation_properties['suggestion_id'][i]
                
                if simuset.simulations[i].exit_code != 0:
                    self.study.clear_suggestion(sid, 'Simulation failed.')
                    self.logger.warn('Simulation with suggestion_id {} failed. Ignoring and continuing...'.format(sid))
                    continue
                
                observed_result = objective_func(simuset.simulations[i]);
                if observed_result is None:
                    self.study.clear_suggestion(sid, 'Simulation skipped by client.')
                    continue
                
                observation = self.study.new_observation()
                observation.add(observed_result)
                self.study.add_observation(observation, sid)
                
        self.logger.info('Finished optimization.')
            
    def __clear_storage_dir(self, simuset):        
        dbase_name = _config.get('DEFAULTS', 'database_name')
        database_file = os.path.join(simuset.storage_dir, dbase_name)
        
        os.remove(database_file)
