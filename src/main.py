import pandas as pd # for data manipulation 
import networkx as nx # for drawing graphs
import matplotlib.pyplot as plt # for drawing graphs

# for creating Bayesian Belief Networks (BBN)
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
import random
from copy import deepcopy

from Helper import *


# hypperparameters: 
num_people = 4
h_threshold = 0.5

# Create Network
bbn = BBN_defnition()

visualize_BBN(bbn)

child_perception_of_people = create_child_perception_of_people(bbn,n=num_people)