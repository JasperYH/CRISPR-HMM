from .hmm_alignment import *
import matplotlib.pyplot as plt
from Bio import SeqIO
import regex
import os




def linear_prior(parameter,MAX,MIN,MAX_POS,length):
    """
    Generate linear prior for a transition probability of the following parameters:
    delta, tau, gamma, pi. Persumbly, the transition probability is highest at the 
    CRISPR induced DSB site and decreases as it moves further.
    
    :param str parameter: Which of the following parameter: delta, tau, gamma, pi.
    
    :param float MAX: The maximum probability.
    
    :param float MIN: The minimum probability.
    
    :param int MAX_POS: The position of reference sequence where the transition 
    probability is the highest. e.g. 4bp upstream to the PAM.
    
    :param int length: The length of the reference sequence.
    
    :return: A list transition probability from the first position to the last position.
    :rtype: list[float]
    """
    
    if parameter == "delta":
        if MAX_POS >= length:
            raise ValueError("MAX_POS of Delta needs to be less than %d" % length)
        s = 0
        l = length
        score = []

    elif parameter == "tau" or parameter == "gamma":
        if MAX_POS >= length + 1:
            raise ValueError("MAX_POS of Tau and Gamma needs to be less than %d" % (length+1))
        s = 0
        l = length + 1
        score = []
        
    elif parameter == "epsilon":
        if MAX_POS == 0 or  MAX_POS>= length:
            raise ValueError("MAX_POS of Epsilon needs to be greater than 0 and less than %d" % length)
        s = 1
        l = length
        score = [0]
    
    if MAX_POS - s != 0:
        b0 = (MAX-MIN)/(MAX_POS-s)
    else:
        b0 = 0
        
    if l-MAX_POS-1 != 0:
        b1 = (MAX-MIN)/(l-MAX_POS-1)
    else:
        b1 = 0
        
    for i in range(s,l):
        if i <= MAX_POS:
            score.append(MAX - b0 * (MAX_POS - i))
        else:
            score.append(MAX - b1 * (i - MAX_POS))  
    for i in range(l,length+1):
        score.append(0)
    return score




def get_all_alignment(s,t):
    """
    Iteratively generate all alignment outcome between sequence s and t.
    
    :param str s: The first sequence.
    
    :param str t: The second sequence.
    
    :return: A list of all alignment outcome.
    :rtype: list[str]
    """
    
    aln_list = [("","")]
    i = 0
    while True:
        tmp = []
        counter = 0
        for a in aln_list:
            i, j = len(a[0]) - a[0].count("-"), len(a[1]) - a[1].count("-")
            if i == len(s) and j == len(t):
                tmp.append(a)
                counter += 1
            else:
                if i < len(s):
                    tmp.append((a[0]+s[i],a[1]+"-"))
                if j < len(t):
                    tmp.append((a[0]+"-",a[1]+t[j]))
                if i < len(s) and j < len(t):
                    tmp.append((a[0]+s[i],a[1]+t[j]))
        if counter == len(aln_list):
            break
        else:
            aln_list = tmp
    return aln_list




def best_aln_by_iteration(model,t):
    """
    Iteratively generate all alignment outcome between sequence s and t.
    
    :param object model: A hmm_model object.
    
    :param str t: A mutated sequence.
    
    :return: The best alignment between the model.reference_sequence and t.
    :rtype: tuple
    """
  
    best_score = -float("inf")
    for aln in get_all_alignment(model.reference_sequence,t):
        score = model.get_likelihood(aln[0],aln[1],use_log=True)
        if score > best_score:
            best_score = score
            best_aln = aln
    return best_aln[0],best_aln[1],best_score




def find_all_alignment_of_state(aln_list,state,pos):
    """
    Find all alignment between sequence s and t containing the 
    hidden state specified by state and position pos.
    
    :param list aln_list: .
    
    :param str t: The second sequence.
    
    :return: A list of all alignment outcome.
    :rtype: list
    """
    
    if state == "S":
        return aln_list, [0 for _ in aln_list]
    
    if state == "E":
        return aln_list, [len(a[0]) for a in aln_list]
    
    state_list = []
    idx = []
    for a in aln_list:
        i = 0
        for k in range(len(a[0])):
            if a[0][k] != "-":
                i += 1
                if state == "M" and pos == i and a[0][k] != "-" and a[1][k] != "-":
                    state_list.append(a)
                    idx.append(k)
                    break
                elif state == "D" and pos == i and a[0][k] != "-" and a[1][k] == "-":
                    state_list.append(a)
                    idx.append(k)
                    break
            else:
                if state == "I" and pos == i and a[0][k] == "-" and a[1][k] != "-":
                    state_list.append(a)
                    idx.append(k)
                    break
    return state_list,idx




def calculate_psi_by_iteration(model,t,state,pos):
    """
    Calculate the psi (marginal probability) of a particular
    state by summing over the the possible alignment through iteration.
    
    :param object model: A hmm_model object.
    
    :param str t: The second sequence.
    
    :param str state: The hidden state of psi. Possibilities are: M, I and D.
    
    :param int pos: The position of hidden state.
    
    :return: The marginal probability of psi(state, pos).
    :rtype: float
    """
    
    aln_list = get_all_alignment(model.reference_sequence,t)
    my_alignment,_ = get_all_alignment_of_state(aln_list,state,pos)
    
    if state == "S":
        prob = 0
        for aln in aln_list:
            prob += model.get_likelihood(aln[0],aln[1])
    elif state == "M" or state == "D":
        prob = 0
        for aln in my_alignment:
            prob += model.get_likelihood(aln[0],aln[1])
    elif state == "I":
        prob = 0
        for aln in my_alignment:
            j = 0
            k = 0
            for i in range(len(aln[0])):
                if aln[0][i] != "-":
                    j += 1
                if j == pos and aln[0][i] == "-":
                    k += 1
            prob += k * model.get_likelihood(aln[0],aln[1])
    return prob




def get_xi_by_iteration(model,t,state1,pos1,state2,pos2):
    """
    Calculate the xi (marginal probability) of a particular state transition
    (state1,pos1) -> (state2,pos2) by summing over the the possible alignment 
    through iteration.
    
    :param object model: A hmm_model object.
    
    :param str t: The second sequence.
    
    :param str state1: The first hidden state. Possibilities are: M, I and D.
    
    :param int pos1: The position of the first hidden state.
    
    :param str state2: The second hidden state. Possibilities are: M, I and D.
    
    :param int pos2: The position of the second hidden state.
    
    :return: The marginal transition probability from (state1, pos1) to
    (state2, pos2).
    :rtype: float
    """
  
    prob = 0
    aln_list = get_all_alignment(model.reference_sequence,t)
    
    if state1 == "S":
        if state2 == "M" and pos2 == 1:
            my_alignment2 = [aln for aln in aln_list if (aln[0][0]!="-" and aln[1][0]!="-")]
        elif state2 == "D" and pos2 == 1:
            my_alignment2 = [aln for aln in aln_list if (aln[0][0]!="-" and aln[1][0]=="-")]
        elif state2 == "I" and pos2 == 0:
            my_alignment2 = [aln for aln in aln_list if (aln[0][0]=="-" and aln[1][0]!="-")]
        else:
            my_alignment2 = []
        for aln in my_alignment2:
            prob += model.get_likelihood(aln[0],aln[1],use_end_prob=True)
    elif state1 == "I":
        my_alignment1, idx1 = get_all_alignment_of_state(aln_list,state1,pos1)
        my_alignment2, idx2 = get_all_alignment_of_state(my_alignment1,state2,pos2)
        result = []
        for i in range(len(my_alignment2)):
            aln = my_alignment2[i]
            if state2 == "I" and pos1 == pos2:
                j = idx2[i] + 1
                while j < len(aln[0]) and aln[0][j] == "-":
                    prob += model.get_likelihood(aln[0],aln[1],use_end_prob=True)
                    j += 1
            elif state2 == "M" or state2 == "D" or state2 == "E":
                prob += model.get_likelihood(aln[0],aln[1],use_end_prob=True)  
    else:
        my_alignment1, idx1 = get_all_alignment_of_state(aln_list,state1,pos1)
        my_alignment2, idx2 = get_all_alignment_of_state(my_alignment1,state2,pos2)
        for i in range(len(my_alignment2)):
            aln = my_alignment2[i]
            j = my_alignment1.index(aln)
            if idx1[j] + 1 == idx2[i]:
                prob += model.get_likelihood(aln[0],aln[1],use_end_prob=True)              
    return prob




def plot_params(model,params=["delta","tau","epsilon","gamma"],colors=["gray","gray","gray","gray"]):
    """
    Plot the transition probability of a model.
    
    :param object model: A hmm_model object.
    
    :param str t: The second sequence.
    
    :param list params: A list of parameters to plot. Default: ["delta","tau","epsilon","gamma"].
    
    :param list colors: A list of colors to use. Default: ["gray","gray","gray","gray"].
    
    :return: Figures of transition probability.
    :rtype: figure
    """
    
    fig, axs = plt.subplots(len(params), 1)
    for i in range(4):
        param = params[i]
        pos = [j for j in range(model.n+1)]
        axs[i].bar(pos, getattr(model,param),color=colors[i])
        if i == 3:
            bar = ["0"] + [j for j in model.reference_sequence]
            axs[i].set_xticks(pos, bar)
        else:
            axs[i].set_xticks([])
        axs[i].set_ylabel("%s"%param,fontsize=10)
    return fig

