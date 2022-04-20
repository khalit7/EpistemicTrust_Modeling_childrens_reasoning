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

''' Helper functions '''
def BBN_defnition(k_prob,h_prob):
  K = BbnNode(Variable(0, 'K', ['1', '0']), [k_prob, 1-k_prob])

  S = BbnNode(Variable(1, 'S', ['A', 'B', 'C','D','E']), [0.2,0.2,0.2,0.2,0.2])

  B = BbnNode(Variable(2, 'B', ['A', 'B', 'C','D','E']), [1,0,0,0,0,
                                                          0.2,0.2,0.2,0.2,0.2,
                                                              0,1,0,0,0,
                                                          0.2,0.2,0.2,0.2,0.2,
                                                              0,0,1,0,0,
                                                          0.2,0.2,0.2,0.2,0.2,
                                                              0,0,0,1,0,
                                                          0.2,0.2,0.2,0.2,0.2,
                                                              0,0,0,0,1,
                                                              0.2,0.2,0.2,0.2,0.2
                                                              ])

  H = BbnNode(Variable(3, 'H', ['1','0']), [h_prob,1-h_prob])

  L = BbnNode(Variable(4, 'L', ['A', 'B', 'C','D','E']), [
                                                              1,0,0,0,0,
                                                              0,0.25,0.25,0.25,0.25,
                                                              0,1,0,0,0,
                                                              0.25,0,0.25,0.25,0.25,
                                                              0,0,1,0,0,
                                                              0.25,0.25,0,0.25,0.25,
                                                              0,0,0,1,0,
                                                              0.25,0.25,0.25,0,0.25,
                                                              0,0,0,0,1,
                                                              0.25,0.25,0.25,0.25,0,                                                         
                                                              ]
              )
  # Create Network
  bbn = Bbn() \
      .add_node(K) \
      .add_node(S) \
      .add_node(B) \
      .add_node(H) \
      .add_node(L) \
      .add_edge(Edge(K, B, EdgeType.DIRECTED)) \
      .add_edge(Edge(S, B, EdgeType.DIRECTED)) \
      .add_edge(Edge(H, L, EdgeType.DIRECTED)) \
      .add_edge(Edge(B, L, EdgeType.DIRECTED)) 

  return bbn

def visualize_BBN(bbn):
  # Set node positions
  pos = {0: (0, 0.5), 1: (-0.5, 0), 2: (0, 0), 3: (0.5, 0), 4:(0,-0.5)}

  # Set options for graph looks
  options = {
      "font_size": 10,
      "node_size": 4000,
      "node_color": "white",
      "edgecolors": "black",
      "edge_color": "red",
      "linewidths": 5,
      "width": 5,}
      
  # Generate graph
  n, d = bbn.to_nx_graph()
  nx.draw(n, with_labels=True, labels=d, pos=pos, **options)

  # Update margins and print the graph
  ax = plt.gca()
  ax.margins(0.10)
  plt.axis("off")
  plt.show()

def create_child_perception_of_people(bbn,n=1):
  people = []
  for i in range(n):
   people.append (InferenceController.apply(bbn[i]))

  return people



# Print marginal probabilities
def print_probs(join_tree):
    for node in join_tree.get_bbn_nodes():
        potential = join_tree.get_bbn_potential(node)
        print("Node:", node)
        print("Values:")
        print(potential)
        print('----------------')

# get Knowledge and helpfulness from each person (as percieved by the child)
def get_infered_k_h(join_tree):
  results = {}
  for node, posteriors in join_tree.get_posteriors().items():
    if node == "K":
      results['K'] = list(posteriors.values())[0]
    if node == "H":
      results['H'] = list(posteriors.values())[0]
  return (results['K'],results['H'])



# To add evidence of events that happened so probability distribution can be recalculated
def evidence(ev, nod, cat, val,join_tree):
    ev = EvidenceBuilder() \
    .with_node(join_tree.get_bbn_node_by_name(nod)) \
    .with_evidence(cat, val) \
    .build()
    join_tree.set_observation(ev)
    return join_tree.get_posteriors().items()

# updates priors depending on posterior
def update_belief(posteriors,join_tree):
  k_h_belief = {}
  for n, p in posteriors:
    if n == "K" or n =="H":
      node_id = join_tree.get_bbn_node_by_name(n).id
      updated_belief = list(p.values())
      k_h_belief[node_id] = updated_belief
  return InferenceController.reapply(join_tree, k_h_belief)

# create people with different knowledge and helpfulness
def create_people(num_people,K_H=None,K_NH=None,NK_H=None,NK_NH=None):
  people = {}
  if K_H is not None:
        idx=0
        for _ in range(K_H):
              people[idx] = (1,1)
              idx+=1
        for _ in range(K_NH):
              people[idx] = (1,0)
              idx+=1
        for _ in range(NK_H):
              people[idx] = (0,1)
              idx+=1
        for _ in range(NK_NH):
              people[idx] = (0,0)
              idx+=1
        return people
  # else
  for i in range(num_people):
    knowledge = random.randint(0, 1)
    helpful = random.randint(0, 1)
    people[i] = (knowledge,helpful)

  return people

def present_random_object(objects):
  idx = random.randint(0, len(objects)-1)

  return objects[idx]