# congress-political-spectrum

![spectrum plots](data/spectrum.png?raw=true)

Visualize the ideological position of members of US Congress on a liberal/conservative spectrum, based on voting record.

## Install & Run
1. Clone this repo `git clone https://github.com/davhbrown/congress-political-spectrum.git`
2. Create a new python environment if you wish, this was developed in python 3.9.12. Python 3.11+ is not supported.
3. `pip install -r requirements.txt`
4. `cd ./congress-political-spectrum`

#### Run with Bokeh server locally
From the command line:
`bokeh serve --show simple_spectrum.py`

This will open a web browser and display the plots.


## Data Source:
Lewis, Jeffrey B., Keith Poole, Howard Rosenthal, Adam Boche, Aaron Rudkin, and Luke Sonnet (2024). Voteview: Congressional Roll-Call Votes Database. https://voteview.com/
