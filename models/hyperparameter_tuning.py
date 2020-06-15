import os
import sys
sys.path.append(os.path.abspath(".."))
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt.pyll.stochastic import sample
from functools import partial
from glob import glob
import re
import numpy as np
from utils.data_handler import read_pickle, save_to_pickle

def extract_trial_results(x, space):
    experiment = space_eval(
        space, {k: (v[0] if v else 0) for k, v in x["misc"]["vals"].items()}
    )
    experiment["num_epochs"] = np.argmax(x["result"]["model_history"]["val_f1_metric"])

    experiment["f1_train"] = x["result"]["model_history"]["f1_metric"][
        experiment["num_epochs"]
    ]
    experiment["f1_val"] = x["result"]["model_history"]["val_f1_metric"][
        experiment["num_epochs"]
    ]
    return experiment


class safeHyperopt:
    def __init__(
        self,
        model,
        space,
        version,
        model_name,
        total_trials=50,
        init_max_trials=3,
        export_directory="../../../models/"
        
    ):

        self.model = model
        self.space = space
        self.init_max_trials = init_max_trials
        self.total_trials = total_trials
        self.export_directory = export_directory
        self.model_name = model_name
        self.version = str(version)
        self.full_export_directory = os.path.join(
            self.export_directory, self.model_name, f"V{self.version}"
        )

        if not os.path.exists(self.full_export_directory):
            os.makedirs(self.full_export_directory)

        save_to_pickle(
            self.space,
            os.path.join(self.full_export_directory, "space_hyperopt.pickle"),
        )

    def get_latest_trial_version(self, path):
        latest_version = 0
        versions = list(map(lambda x: int(re.findall("\d", x)[0]), glob(path)))
        if versions:
            latest_version = max(versions)
        return latest_version

    def run_trials(self, verbose: bool = True):

        trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
        n_startup_jobs = 2

        try:  # try to load an already saved trials object, and increase the max
            trials = read_pickle(
                os.path.join(self.full_export_directory, "trials.pickle")
            )
            print("Found saved Trials! Loading...")
            self.init_max_trials = len(trials.trials) + trials_step

            best_loss = trials.best_trial["result"]["loss"]

            print(
                "Rerunning from {} trials to {} (+{}) trials".format(
                    len(trials.trials), self.init_max_trials, trials_step
                )
            )
        except:  # create a new trials object and start searching
            trials = Trials()

        algo = partial(tpe.suggest, n_startup_jobs=n_startup_jobs)
        best = fmin(
            self.model,
            self.space,
            algo=algo,
            max_evals=self.init_max_trials,
            trials=trials,
            verbose=verbose,
        )

        print("Best:", best)

        # save the trials object
        self.current_num_trials = len(trials.trials)
        save_to_pickle(
            trials, os.path.join(self.full_export_directory, "trials.pickle")
        )

    def run_train_loop(self):

        try:  # try to load an already saved trials object, and increase the max
            trials = read_pickle(
                os.path.join(self.full_export_directory, "trials.pickle")
            )
            self.current_num_trials = len(trials.trials)
        except:
            self.current_num_trials = 0

        while True:

            print("NUM TRIALS", self.current_num_trials)
            if self.current_num_trials < self.total_trials:
                self.run_trials()
            else:
                print("Total trials reaches!")
                break
