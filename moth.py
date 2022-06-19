from __future__ import annotations
from __future__ import division, print_function, unicode_literals

import sys
import time  # output some timings per evaluation
from collections import defaultdict
import os, webbrowser  # to show post-processed results in the browser
import cocoex  # experimentation module

try:
    import cocopp  # post-processing module
except:
    pass
import numpy as np
import cma

### MKL bug fix
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

fmin = cma.fmin2
suite_name = "bbob-biobj"  # see cocoex.known_suite_names
budget_multiplier = 2  # times dimension, increase to 10, 100, ...
suite_filter_options = (""  # without filtering, a suite has instance_indices 1-15
                        # "dimensions: 2,3,5,10,20 "  # skip dimension 40
                        # "instance_indices: 1-5 "  # relative to suite instances
                        )
# for more suite filter options see http://numbbo.github.io/coco-doc/C/#suite-parameters
suite_year_option = ""  # "year: 2022"  # determine instances by year, not all years work for all suites :-(

batches = 1  # number of batches, batch=3/32 works to set both, current_batch and batches
current_batch = 1  # only current_batch modulo batches is relevant
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
# observer_cma = cocoex.Observer(suite_name, "result_folder: " + output_folder)
minimal_print = cocoex.utilities.MiniPrint()
stoppings = defaultdict(list)  # dict of lists, key is the problem index
timings = defaultdict(list)  # key is the dimension

stoppings_cma = defaultdict(list)  # dict of lists, key is the problem index
timings_cma = defaultdict(list)  # key is the dimension
### go
print('*** benchmarking %s from %s on suite %s ***'
      % (fmin.__name__, fmin.__module__, suite_name))
time0 = time.time()

for batch_counter, problem in enumerate(suite):  # this loop may take hours or days...
    if batch_counter % batches != current_batch % batches:
        continue
    if not len(timings_cma[problem.dimension]) and len(timings_cma) > 1:
        print("\n   %s %d-D done in %.1e seconds/evaluations"
              % (minimal_print.stime, sorted(timings_cma)[-2],
                 np.median(timings_cma[sorted(timings_cma)[-2]])), end='')
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
        xopt, es = fmin(func, propose_x0, 2,
                        {'maxfevals': evalsleft(), 'verbose': -9}, restarts=9)
        stoppings_cma[problem.index].append(es.stop())

    timings_cma[problem.dimension].append((time.time() - time1) / problem.evaluations
                                          if problem.evaluations else 0)
    minimal_print(problem, restarted=irestart, final=problem.index == len(suite) - 1)
    with open(output_folder + '_stopping_conditions.pydict', 'wt') as file_:
        file_.write("# code to read in these data:\n"
                    "# import ast\n"
                    "# with open('%s_stopping_conditions.pydict', 'rt') as file_:\n"
                    "#     stoppings_cma = ast.literal_eval(file_.read())\n"
                    % output_folder)
        file_.write(repr(dict(stoppings_cma)))

### print timings and final message

print("\n   %s %d-D done in %.1e seconds/evaluations"
      % (minimal_print.stime, sorted(timings_cma)[-1], np.median(timings_cma[sorted(timings_cma)[-1]])))
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

for dimension in sorted(timings_cma):
    print("    %3d       %.1e" % (dimension, np.median(timings_cma[dimension])))
print("  -------------------------------------")

### post-process data
if batches == 1 and 'cocopp' in globals() and cocopp not in (None, 'None'):
    cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
    webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")
