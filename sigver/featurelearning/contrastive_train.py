import argparse
import warnings
import sigver.featurelearning.first_task as first_task
import sigver.featurelearning.second_task as second_task
import sys


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    argparser = argparse.ArgumentParser('Train Contrastive Model')
    argparser.add_argument('--task', help='Single-task or multi-task', choices=('multi','single'), default='multi', type=str)
    arguments, unknown = argparser.parse_known_args()
    print(arguments)

    if arguments.task == 'multi':
        print('Model is being trained in a Multi-task approach..')
        first_task.build_arguments()
        second_task.build_arguments()
    else:
        print('Model is being trained in a Single-task approach..')
        second_task.build_arguments()

