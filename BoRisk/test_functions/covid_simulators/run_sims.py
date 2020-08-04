import sys
import yaml
import time
import os
import multiprocessing
from BoRisk.test_functions.covid_simulators.analysis_helpers import \
    run_multiple_trajectories
import cPickle as pickle

BASE_DIRECTORY="/nfs01/covid_sims/"

def run_background_sim(output_dir, sim_params, ntrajectories=150, time_horizon=112):
    try:
        dfs = run_multiple_trajectories(sim_params, ntrajectories, time_horizon)
        
        # record output
        for idx, df in enumerate(dfs):
            df_file_name = "{}/{}.csv".format(output_dir, idx)
            df.to_csv(df_file_name)
    except Exception as e:
        error_msg = "Encountered error: {}".format(str(e))
        print(error_msg)
        f = open(output_dir + "/error.txt", "w")
        f.write(error_msg)
        f.close()

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python {} yaml-config-file ('fall' or 'base') (optional:simulation-name)".format(sys.argv[0]))
        exit()

    if sys.argv[2] == 'fall':
        from fall_params import base_params
    elif sys.argv[2] == 'base':
        from base_params import base_params
    else:
        print("Error: second argument must be 'fall' or 'base', but got {}".format(sys.argv[2]))

    sim_config = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)

    params = base_params.copy()
    if 'base_params_to_update' in sim_config and sim_config['base_params_to_update'] != None:
        for param, val in sim_config['base_params_to_update'].items():
            if param not in params and param != 'contact_trace_testing_frac':
                print("Configuration attempting to modify non-existent parameter {}".format(
                    param))
                exit()
            else:
                params[param] = val

    sim_timestamp = time.time()
    if len(sys.argv) >= 3:
        sim_id = "{}.{}".format(sys.argv[3], sim_timestamp)
    else:
        sim_id = str(sim_timestamp)
    print("Simulation ID: {}".format(sim_id))
    sim_main_dir = BASE_DIRECTORY + str(sim_id)
    try: 
        os.mkdir(sim_main_dir)
        print("Output directory {} created".format(sim_main_dir))
    except FileExistsError:
        print("Output directory {} already exists".format(sim_main_dir))
        exit()

    
    param_to_vary = sim_config['param_to_vary']
    param_values = sim_config['parameter_values']
    if 'ntrajectories' in sim_config:
        ntrajectories = sim_config['ntrajectories']
    else:
        ntrajectories = 150

    if 'time_horizon' in sim_config:
        time_horizon = sim_config['time_horizon']
    else:
        time_horizon = 112 # 16 weeks

    if len(param_values) == 0:
        print("Empty list of parameters given; nothing to do")
        exit()

    for param_val in param_values:
        # create the relevant subdirectory
        sim_sub_dir = "{}/{}.{}".format(sim_main_dir, param_to_vary, param_val)
        os.mkdir(sim_sub_dir)
        print("Created directory {} to save output".format(sim_sub_dir))
        # instantiate relevant sim params
        sim_params = params.copy()
        sim_params[param_to_vary] = param_val
        pickle.dump(sim_params, open("{}/sim_params.pickle".format(sim_sub_dir), "wb"))
        print("Saved sim_params to pickle file")
        # start new process
        fn_args = (sim_sub_dir, sim_params, ntrajectories, time_horizon)
        proc = multiprocessing.Process(target = run_background_sim, args=fn_args)
        #proc.daemon = True
        proc.start()
        print("starting process for {} value {}".format(param_to_vary, param_val))
        print("process PID = {}".format(proc.pid))

    print("Waiting for processes to finish...")

        








