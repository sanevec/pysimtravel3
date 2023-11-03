# PySimtravel 3

## Paper1 - Integrating Efficient Routes with Station Monitoring for EVs

This tool allows you to manage and run scientific experiments in a selective manner. You can list all experiments, execute a specific one, run all experiments in the background, or generate meta statistics based on the results.

### Usage

The program is executed from the command line using Python 3. Below are the available options:

''' bash
python3 paper1.py -h
'''

This will display the following help message with all the available options:

''' bash
usage: paper1.py [-h] [--list] [--run RUN] [--all] [--stats]

Selectively run experiments.

options:
-h, --help show this help message and exit
--list List all experiments
--run RUN Run a specific experiment by index
--all Run all experiments in the background
--stats Generate meta statistics
'''

### Examples

- To list all experiments:

''' bash
python3 paper1.py --list
'''

- To run the experiment with a specific index (e.g., index 42):

''' bash
python3 paper1.py --run 42
'''

- To run all experiments in the background:

''' bash
python3 paper1.py --all
'''

- To generate the meta statistics:

''' bash
python3 paper1.py --stats
'''
