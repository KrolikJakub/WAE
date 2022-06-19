from __future__ import annotations
from __future__ import division, print_function, unicode_literals

import sys
import time  # output some timings per evaluation
from collections import defaultdict
import os, webbrowser  # to show post-processed results in the browser
import numpy as np  # for median, zeros, random, asarray
import cocoex  # experimentation module

try:
    import cocopp  # post-processing module
except:
    pass

# EGO
import trieste
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition import (
    SingleModelAcquisitionBuilder,
    ExpectedImprovement,
    Product,
)
from trieste.space import Box
from trieste.models.gpflow import build_gpr, build_vgp_classifier
from trieste.models import TrainableProbabilisticModel
from trieste.models.gpflow.models import (
    GaussianProcessRegression,
    VariationalGaussianProcess,
)
from trieste.models.optimizer import BatchOptimizer
import numpy as np
import tensorflow as tf


# MKL bug fix
def set_num_threads(nt=1, disp=1):
    """see https://github.com/numbbo/coco/issues/1919
    and https://twitter.com/jeremyphoward/status/1185044752753815552
    """
    try:
        import mkl
    except ImportError:
        disp and print("mkl is not installed")
    else:
        mkl.set_num_threads(nt)
    nt = str(nt)
    for name in ['OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS',
                 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS']:
        os.environ[name] = nt
    disp and print("setting mkl threads num to", nt)


if sys.platform.lower() not in ('darwin', 'windows'):
    set_num_threads(1)


fmin = trieste.bayesian_optimizer.BayesianOptimizer.optimize
suite_name = "bbob-biobj"
budget_multiplier = 2
suite_filter_options = ("")
suite_year_option = ""

batches = 1
current_batch = 1
output_folder = ''

### possibly modify/overwrite above input parameters from input args
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ('-h', 'help', '-help', '--help'):
        print(__doc__)
        raise ValueError("printed help and aborted")
    input_params = cocoex.utilities.args_to_dict(
        sys.argv[1:], globals(), {'batch': 'current_batch/batches'}, print=print)
    globals().update(input_params)  # (re-)assign variables

# extend output folder input parameter, comment out if desired otherwise
output_folder += '%s_of_%s_%dD_on_%s' % (
    fmin.__name__, fmin.__module__, int(budget_multiplier), suite_name)

if batches > 1:
    output_folder += "_batch%03dof%d" % (current_batch, batches)

### prepare
suite = cocoex.Suite(suite_name, suite_year_option, suite_filter_options)
observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
minimal_print = cocoex.utilities.MiniPrint()
stoppings = defaultdict(list)  # dict of lists, key is the problem index
timings = defaultdict(list)  # key is the dimension

### go
print('*** benchmarking %s from %s on suite %s ***'
      % (fmin.__name__, fmin.__module__, suite_name))
time0 = time.time()
for batch_counter, problem in enumerate(suite):  # this loop may take hours or days...
    if problem.dimension == 3: break
    if batch_counter % batches != current_batch % batches:
        continue
    if not len(timings[problem.dimension]) and len(timings) > 1:
        print("\n   %s %d-D done in %.1e seconds/evaluations"
              % (minimal_print.stime, sorted(timings)[-2],
                 np.median(timings[sorted(timings)[-2]])), end='')
    problem.observe_with(observer)  # generate the data for cocopp post-processing
    problem(np.zeros(problem.dimension))  # making algorithms more comparable
    propose_x0 = problem.initial_solution_proposal  # callable, all zeros in first call
    evalsleft = lambda: int(problem.dimension * budget_multiplier + 1 -
                            max((problem.evaluations, problem.evaluations_constraints)))
    time1 = time.time()
    func = lambda x: sum(problem(x))
    irestart = -1
    while evalsleft() > 0 and not problem.final_target_hit:
        irestart += 1
        # EGO EGO EGO EGO

        np.random.seed(1234)
        tf.random.set_seed(1234)

        search_space = Box([-100, -100], [100, 100])
        OBJECTIVE = "OBJECTIVE"
        FAILURE = "FAILURE"


        def observer_xd(x):
            results = []
            for arguments in x:
                results.append([func(arguments)])
            y = np.array(results)
            mask = np.isfinite(y).reshape(-1)
            return {
                OBJECTIVE: trieste.data.Dataset(x[mask], y[mask]),
                FAILURE: trieste.data.Dataset(x, tf.cast(np.isfinite(y), tf.float64)),
            }


        num_init_points = 15
        initial_data = observer_xd(search_space.sample(num_init_points))

        regression_model = build_gpr(
            initial_data[OBJECTIVE], search_space, likelihood_variance=1e-7
        )
        classification_model = build_vgp_classifier(
            initial_data[FAILURE], search_space, noise_free=True
        )

        models: dict[str, TrainableProbabilisticModel] = {
            OBJECTIVE: GaussianProcessRegression(regression_model),
            FAILURE: VariationalGaussianProcess(
                classification_model,
                BatchOptimizer(tf.optimizers.Adam(1e-3)),
                use_natgrads=True,
            ),
        }


        class ProbabilityOfValidity(SingleModelAcquisitionBuilder):
            def prepare_acquisition_function(self, model, dataset=None):
                def acquisition(at):
                    mean, _ = model.predict_y(tf.squeeze(at, -2))
                    return mean

                return acquisition


        ei = ExpectedImprovement()
        pov = ProbabilityOfValidity()
        acq_fn = Product(ei.using(OBJECTIVE), pov.using(FAILURE))
        rule = EfficientGlobalOptimization(acq_fn)  # type: ignore

        bo = trieste.bayesian_optimizer.BayesianOptimizer(observer_xd, search_space)

        num_steps = 15
        result = bo.optimize(
            num_steps, initial_data, models, rule
        ).final_result.unwrap()

        arg_min_idx = tf.squeeze(
            tf.argmin(result.datasets[OBJECTIVE].observations, axis=0)
        )
        print(f"query point: {result.datasets[OBJECTIVE].query_points[arg_min_idx, :]}")

    timings[problem.dimension].append((time.time() - time1) / problem.evaluations
                                      if problem.evaluations else 0)
    minimal_print(problem, restarted=irestart, final=problem.index == len(suite) - 1)
    with open(output_folder + '_stopping_conditions.pydict', 'wt') as file_:
        file_.write("# code to read in these data:\n"
                    "# import ast\n"
                    "# with open('%s_stopping_conditions.pydict', 'rt') as file_:\n"
                    "#     stoppings = ast.literal_eval(file_.read())\n"
                    % output_folder)
        file_.write(repr(dict(stoppings)))

if batches > 1:
    print("*** Batch %d of %d batches finished in %s."
          " Make sure to run *all* batches (via current_batch or batch=#/#) ***"
          % (current_batch, batches, cocoex.utilities.ascetime(time.time() - time0)))
else:
    print("*** Full experiment done in %s ***"
          % cocoex.utilities.ascetime(time.time() - time0))

print("Timing summary:\n"
      "  dimension  median seconds/evaluations\n"
      "  -------------------------------------")
for dimension in sorted(timings):
    print("    %3d       %.1e" % (dimension, np.median(timings[dimension])))
print("  -------------------------------------")

### post-process data
if batches == 1 and 'cocopp' in globals() and cocopp not in (None, 'None'):
    cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
    webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")
