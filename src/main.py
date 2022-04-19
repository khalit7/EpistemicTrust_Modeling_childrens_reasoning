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



#################################################################################################################
num_people = 4
bbn = BBN_defnition()
child_perception_of_people = create_child_perception_of_people(bbn,n=num_people)
people = create_people(num_people)

h_threshold = 0.5

# child's knowldege about the world
child_knowldge = {0:[0.2,0.2,0.2,0.2,0.2],
                  1:[0.2,0.2,0.2,0.2,0.2],
                  2:[0.2,0.2,0.2,0.2,0.2],
                  3:[0.2,0.2,0.2,0.2,0.2],
                  4:[0.2,0.2,0.2,0.2,0.2],
                  }
                  
idx_to_object = {0:'A',1:'B',2:'C',3:'D',4:'E'}
object_to_idx = {v:k for k,v in idx_to_object.items()}

people = {0:(1,0),1:(1,1),2:(1,1),3:(1,0)}


def vote(presented_obj,people):
  votes = {}
  for i in range(len(people)):
    knowledge,helpful = people[i]
    objects = list(idx_to_object.values())
    if knowledge == 1 and helpful == 1:
      votes[i] = presented_obj
    elif knowledge == 1 and helpful == 0:
      objects.remove(presented_obj)
      votes[i] = present_random_object(objects)
    elif knowledge == 0:
      votes[i] = present_random_object(objects)
  return votes


def choose_label(votes,presented_obj_idx):
  result = [0]*len(child_knowldge)
  for i,ch in enumerate(child_perception_of_people):
    k,h = get_infered_k_h(ch)
    if  h >= h_threshold-0.1: # helpful
      result [ object_to_idx [votes[i]] ] += k


  summation = sum(result)
  for i in range(len(result)):
    result[i] /= summation

  for i in range(len(result)):
    result[i] = result [i] + child_knowldge[presented_obj_idx][i]

  summation = sum(result)
  for i in range(len(result)):
    result[i] /= summation

  # update child knowledge
  child_knowldge[presented_obj_idx] = result
  # final label

  max_idx = result.index(max(result))
  label = idx_to_object[max_idx] 
  return label



def update_all_people_belief(votes,s):
  result = []
  child_percption_copy = deepcopy(child_perception_of_people)
  for i in range(num_people):
    child_percption_copy[i] = InferenceController.reapply(child_percption_copy[i], {1:s})
    posteriors = evidence('', 'L', votes[i], 1.0,child_percption_copy[i])
 
    child_perception_of_people[i] = update_belief(posteriors,child_percption_copy[i])
    result.append(get_infered_k_h(child_perception_of_people[i]))
  print("result = ",result)
  return result
        


##############################################################################################################################







# Main loop (where the child interacts with people)
##############################################
num_interactions = 2
#num_interactions = 1
k_h_result_history = []
child_knowldge_history = []
for i in range (num_interactions):

  objects = list(idx_to_object.values())
  presented_obj = present_random_object(objects)
  #presented_obj="A"
  presented_obj_idx = object_to_idx[presented_obj]
  #voting stage
  votes = vote(presented_obj,people)
  # choose label from votes 
  label = choose_label(votes,presented_obj_idx)

  s = child_knowldge[presented_obj_idx]
  # update belief about people
  k_h_result = update_all_people_belief(votes,s)

  k_h_result_history.append(k_h_result)
  child_knowldge_history.append(child_knowldge.copy())
  #print("people = ",people)
  #print("presented object = ",presented_obj)
  #print("votes = ",votes)
  #print("child knowldge : ",child_knowldge)
  
  #print(s)

print("child knowledge",child_knowldge_history[0])
print("child knowledge",child_knowldge_history[1])
