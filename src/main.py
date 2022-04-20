from itertools import count
from threading import local
import pandas as pd # for data manipulation 
import networkx as nx # for drawing graphs
import matplotlib.pyplot as plt # for drawing graphs
from matplotlib import animation
from matplotlib.animation import FuncAnimation,PillowWriter 
import matplotlib.patches as mpatches

# for creating Bayesian Belief Networks (BBN)
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
import random
from copy import deepcopy
import time

from Helper import *
from collections import defaultdict,Counter

random.random
#################################################################################################################
num_interactions = int(input("Please enter the number of interaction the child will make with people : "))
# print("\n Note that people's actual knowledge and intent as well as the child prior about these two variables will be set randomly \n")
decsion = int(input("Enter 1 if you wish to choose how many people belong to each catagory, or press enter to do random assignment : "))
if decsion == 1:
      num_people = None
      K_H=int(input("How many people are knowledgable with good intent ? "))
      K_NH=int(input("How many people are knowledgable with bad intent ? "))
      NK_H=int(input("How many people are not knowledgable with good intent ? "))
      NK_NH=int(input("How many people are not knowledgable with bad intent ? "))
      num_people = K_H+K_NH+NK_H+NK_NH
      print("Total number of people = {}".format(num_people))
else:
  num_people = int(input("Please enter the number of people : "))
  K_H=None
  K_NH=None
  NK_H=None
  NK_NH=None

bbn_list=[]
k_h_priors = []
for i in range(num_people):
  k_prob = random.uniform(0, 1)
  h_prob = random.uniform(0, 1)
  bbn = BBN_defnition(k_prob,h_prob)
  bbn_list.append(bbn)
  k_h_priors.append((k_prob,h_prob))
child_perception_of_people = create_child_perception_of_people(bbn_list,n=num_people)
people = create_people(num_people,K_H=K_H,K_NH=K_NH,NK_H=NK_H,NK_NH=NK_NH)

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
  return result
        


##############################################################################################################################






#people = {0:(1,1),1:(1,1),2:(1,0),3:(1,0)}
# Main loop (where the child interacts with people)
##############################################
# num_interactions = 10
#num_interactions = 1
k_h_result_history = defaultdict(list)
child_knowldge_history = []
for i in range (num_interactions):
  print("{}/{} interactions made".format(i+1,num_interactions))
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
  for i in range(len(k_h_result)):
      k_h_result_history[i].append(k_h_result[i])

  child_knowldge_history.append(child_knowldge.copy())

k_h_result_history = dict(k_h_result_history)

input("\n enter anything to visualize the child's learning curve")

def plot_people_k_h():
    max_plots = 6 #keep this number even
    num_plots = min(num_people,max_plots)
    idx = 0
    fig, axs = plt.subplots(2,num_plots//2)
    axs = axs.flatten()
    def animate(i):
        nonlocal idx,axs
        if idx>num_interactions:
          time.sleep(5)
          plt.close(fig)
        for i in range(num_plots):
            axs[i].cla()
            axs[i].plot(k_h_result_history[i][0:idx])
            axs[i].title.set_text('Knowledgablity = {}, Helpfulness = {}'.format(people[i][0],people[i][1]))
            axs[i].legend( ("Knowledgability (prior = {:0.2})".format(k_h_priors[i][0]),"Helpfulness (prior = {:0.2})".format(k_h_priors[i][1])), loc='upper left', shadow=True )
            axs[i].set_xlabel("number of interactions")
            axs[i].set_ylabel("Child's belief about K and H")
        idx+=1



    ani = FuncAnimation(fig,animate,interval = 100,frames=50)
    fig.suptitle("Child's learning curve for differnet informants intent and knoweldge")
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()
    return ani

def plot_object_learning():
    people_stats = Counter(list(people.values()))
    people_stats = {"K={},H={}".format(k[0],k[1]):v for k,v in people_stats.items()}

    num_plots = 6
    idx = 0
    fig, axs = plt.subplots(2,3)
    axs = axs.flatten()
    #
    green_patch = mpatches.Patch(color='green', label='Correct label')
    red_patch = mpatches.Patch(color='red', label='Wrong label')
    #
    def animate(i):
        nonlocal idx,axs
        if idx>=num_interactions:
            time.sleep(5)
            plt.close(fig)
            return
        for i in range(num_plots-1):
            colors = ["red"]*5
            colors[i] = "green"
            axs[i].cla()
            axs[i].bar(list(idx_to_object.values()),child_knowldge_history[idx][i],color=colors)
            axs[i].title.set_text('Object = {}'.format(idx_to_object[i]))
            axs[i].set_xlabel("Number of interactions = {}".format(idx+1))
            axs[i].set_ylabel("Probaility of each object")

            axs[i].legend(handles=[green_patch,red_patch], loc='upper left', shadow=True )

        
        axs[5].cla()
        axs[5].bar(list(people_stats.keys()),list(people_stats.values()))
        axs[5].title.set_text('informants stats'.format(idx_to_object[i]))
        axs[5].set_ylabel("count")
        axs[5].set_xlabel(" K=1 => Knowledgable, K=0 => Not Knowledgable \n H=1 => Helpful, H=0 => Not Helpful")
        
        ###
        idx+=1


    ani = FuncAnimation(fig,animate,interval = 500,)
    fig.suptitle("Child's learning curve for different objects")
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()
    return ani

ani = plot_people_k_h()
ani = plot_object_learning()


