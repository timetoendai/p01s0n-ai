"""p01s0ning-benchmark runscript creator.

Move the file p01s0n_setups_from_scratch.pickle to the folder p01s0ns or change its address to the location on your machine.
The evaluation assumes that a folder called p01s0ning-benchmark is located on the same level as this framework.
"""


# ######## BREW p01s0nS ####################
file_name = 'p01s0n_brewing'
benchmark_file = f'benchmark_{file_name}.sh'
experiments = []

def _command(idx, name, ensemble, model, add_options):
    setup = f'--name bm_{name} --benchmark p01s0ns/p01s0n_setups_from_scratch.pickle --save benchmark --vruns 0'
    return f'python brew_p01s0n.py {setup} --ensemble {ensemble} --net {model} {add_options} --eps 8 --benchmark_idx {idx}'

with open(benchmark_file, 'w+') as file:
    source_model = 'ResNet18'
    add_options = ''
    ensemble = 1
    experiment_name = 'base'
    experiments.append((experiment_name, source_model))
    for i in range(100):
        file.write(_command(i, experiment_name, ensemble, source_model, add_options) + '\n')

    # source_model = 'ResNet18'
    # add_options = '--gradient_noise 0.001'
    # ensemble = 1
    # experiment_name = 'private'
    # experiments.append((experiment_name, source_model))
    # for i in range(100):
    #     file.write(_command(i, experiment_name, ensemble, source_model, add_options) + '\n')

    source_model = 'ResNet18'
    add_options = ''
    ensemble = 4
    experiment_name = 'ens4'
    experiments.append((experiment_name, source_model))
    for i in range(100):
        file.write(_command(i, experiment_name, ensemble, source_model, add_options) + '\n')

    source_model = 'ResNet18,VGG11,MobileNetV2'
    add_options = '--pbatch 128 --tau 0.5'
    ensemble = 6
    experiment_name = 'het'
    experiments.append((experiment_name, source_model))
    for i in range(100):
        file.write(_command(i, experiment_name, ensemble, source_model, add_options) + '\n')

    # source_model = 'ResNet18'
    # add_options = ''
    # ensemble = 8
    # experiment_name = 'ens8'
    # experiments.append((experiment_name, source_model))
    # for i in range(100):
    #     file.write(_command(i, experiment_name, ensemble, source_model, add_options) + '\n')

# ######## BENCHMARK p01s0nS ################################################
file_name = 'p01s0n_evaluation'
benchmark_file = f'benchmark_{file_name}.sh'

def _command(idx, name, brew_model, eval_model):
    folder_name = f'bm_{name}_{brew_model}'
    setup = f"--p01s0ns_path p01s0ns/benchmark_results/{folder_name}/{idx} --from_scratch --model {eval_model} --output bm_{folder_name}_{eval_model}"
    return f'python ../p01s0ning-benchmark/benchmark_test.py {setup}'

with open(benchmark_file, 'w+') as file:
    for (experiment_name, source_model) in experiments:
        for i in range(100):
            for eval_model in ['ResNet18', 'VGG11', 'MobileNetV2']:
                file.write(_command(i, experiment_name, '_'.join(source_model.split(',')), eval_model) + '\n')


