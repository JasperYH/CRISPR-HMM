import numpy as np
from math import log, e
import pathos.multiprocessing as mp
from .fast_viterbi import fast_viterbi
from functools import partial



class hmm_model(object):
    """
    The class to perform sequence alignment with hidden Markov model.
    
    :param str reference_sequence: a reference sequence.
    
    :param delta: the transition probability from match state to deletion state.
    :type delta: float, list
      
    :param float tau: the transition probability from match state to insertion state.
    :type tau: float, list
    
    :param float epsilon: the transition probability from deletion state to deletion state.
    :type epsilon: float, list
      
    :param float gamma: the transition probability from insertion state to insertion state.
    :type epsilon: float, list
        
    :param float rho: the transition probability from deletion state to insertion state.
      
    :param float pi: the transition probability from insertion state to deletion state.
    
    :param float p: the emission probability of match. The emission probability of
      mismatch is 1-p.
      
    :param float q: the emission probability of insertion and deletion. q should be smaller than p
    """
            
  
    def __init__(self, reference_sequence, delta=0.3, tau=0.3, epsilon=0.6, \
                 gamma=0.6, rho=0.03, pi=0.03, q=0.4, p=0.9):
        self.reference_sequence = reference_sequence
        self.n = len(self.reference_sequence)
        
        if type(delta) == list:
            self.delta = delta
        else:
            self.delta = [delta for _ in range(self.n)] + [0]
            
        if type(tau) == list:
            self.tau = tau
        else:
            self.tau = [tau for _ in range(self.n+1)]           
            
        if type(epsilon) == list:
            self.epsilon = epsilon
        else:
            self.epsilon = [0] + [epsilon for _ in range(self.n-1)] + [0] 
            
        if type(gamma) == list:
            self.gamma = gamma
        else:
            self.gamma = [gamma for _ in range(self.n+1)]
            
        self.rho = rho
        self.pi = pi
        self.q = q
        self.match = p
        self.mismatch = 1 - self.match
        self._check_parameters()
        self.likelihood = 1e-100
    
    
    
    
    def calculate_alignment_probability(self, aligned_ref, aligned_seq, fix_trivial_param=True, use_log=False):
        """
        Calculate the probabity of a alignment.
        
        :param str aligned_ref: The reference sequence after alignment. e.g. A-T
        
        :param str aligned_seq: A mutated sequence after alignment. e.g. AC-
        
        :param boolean use_log: return log of probability.
        
        :return float: The probability of the alignment.
        """
      
        prob = 1
        hidden_state = "S"
        position = 0
        MM_fix = 1-np.array(self.delta)-np.array(self.tau)
        MM_fix = np.mean(MM_fix[MM_fix!=0])
        DM_fix = 1-self.rho-np.array(self.epsilon)
        DM_fix = np.mean(DM_fix[DM_fix!=0])
        IM_fix = 1-self.pi-np.array(self.gamma)
        IM_fix = np.mean(IM_fix[IM_fix!=0])

        for i in range(len(aligned_ref)):
            deltaC = self.delta[position]
            tauC = self.tau[position]
            epsilonC = self.epsilon[position]
            gammaC = self.gamma[position]
            transition = {("S","M"):(1-self.tau[0]-self.delta[0]),("S","D"):self.delta[0],("S","I"):self.tau[0],
                          ("M","M"):(1-tauC-deltaC),("M","D"):deltaC,("M","I"):tauC,
                          ("D","M"):(1-epsilonC-self.rho),("D","D"):epsilonC,("D","I"):self.rho,
                          ("I","M"):(1-gammaC-self.pi),("I","I"):gammaC,("I","D"):self.pi}
            if fix_trivial_param:
                transition = {("S","M"):MM_fix,("S","D"):self.delta[0],("S","I"):self.tau[0],
                              ("M","M"):MM_fix,("M","D"):deltaC,("M","I"):tauC,
                              ("D","M"):DM_fix,("D","D"):epsilonC,("D","I"):self.rho,
                              ("I","M"):IM_fix,("I","I"):gammaC,("I","D"):self.pi}

            if aligned_ref[i] != "-" and aligned_seq[i] != "-" and aligned_ref[i] == aligned_seq[i]:
                prob *= self.match * transition[(hidden_state,"M")]
                hidden_state = "M"
                position += 1
            elif aligned_ref[i] != "-" and aligned_seq[i] != "-" and aligned_ref[i] != aligned_seq[i]:
                prob *= self.mismatch * transition[(hidden_state,"M")]
                hidden_state = "M"
                position += 1
            elif aligned_ref[i] != "-" and aligned_seq[i] == "-":
                prob *= self.q * transition[(hidden_state,"D")]
                hidden_state = "D"
                position += 1
            elif aligned_ref[i] == "-" and aligned_seq[i] != "-":
                prob *= self.q * transition[(hidden_state,"I")]
                hidden_state = "I"

        transition = {("M","E"):(1-self.tau[position]),("I","E"):1-self.gamma[position],("D","E"):1}
        if fix_trivial_param:
            transition = {("M","E"):MM_fix,("I","E"):IM_fix,("D","E"):DM_fix}

        prob *= transition[(hidden_state,"E")]
        if use_log:
            return self._log(prob)
        return prob
    
    
    
    
    def viterbi(self, seqs, fix_trivial_param=True, ncores=1):        
        log_delta = np.log(np.array(self.delta)+1e-60)
        log_tau = np.log(np.array(self.tau)+1e-60)
        log_MM = np.log(1-np.array(self.delta)-np.array(self.tau))
        log_MM[-1] = np.log(1-np.array(self.tau)[-1])
        log_epsilon = np.log(np.array(self.epsilon)+1e-60)
        log_rho = np.log(self.rho)
        log_DM = np.log(1-self.rho-np.array(self.epsilon))
        log_gamma = np.log(np.array(self.gamma)+1e-60)
        log_pi = np.log(self.pi)
        log_IM = np.log(1-self.pi-np.array(self.gamma))
        log_IM[-1] = np.log(1-np.array(self.gamma)[-1])
        
        if fix_trivial_param:
            MM = 1-np.array(self.delta)-np.array(self.tau)
            log_MM[log_MM!=0] = np.log(np.mean(MM[MM!=0]))
            DM = 1-self.rho-np.array(self.epsilon)
            log_DM[log_DM!=0] = np.log(np.mean(DM[DM!=0]))
            IM = 1-self.pi-np.array(self.gamma)
            log_IM[log_IM!=0] = np.log(IM[IM!=0])
        
        log_q = np.log(self.q)
        log_match = np.log(self.match)
        log_mismatch = np.log(1-self.match)

        if ncores == 1:
            for t in seqs:
                r = fast_viterbi(ref=self.reference_sequence, t=t, \
                               delta=log_delta, tau=log_tau, \
                               MM=log_MM, epsilon=log_epsilon, \
                               rho=log_rho, DM=log_DM, \
                               gamma=log_gamma, pi=log_pi, \
                               IM=log_IM, q=log_q,\
                               match=log_match, mismatch=log_mismatch)
                yield r

        else:
            with mp.Pool(ncores) as p:
                viterbi = partial(fast_viterbi, ref=self.reference_sequence, \
                               delta=log_delta, tau=log_tau, \
                               MM=log_MM, epsilon=log_epsilon,
                               rho=log_rho, DM=log_DM, \
                               gamma=log_gamma, pi=log_pi, \
                               IM=log_IM, q=log_q,\
                               match=log_match, mismatch=log_mismatch)
                results = p.imap(viterbi, seqs)
                for r in results:
                    yield r
    
    
    
    
    def Baum_Welch_estimator(self,seqs,method="Constrained_MLE",\
                             param_to_estimate=["delta","tau","epsilon","gamma"],\
                             lower_bound=None,ncores=1,tol=1e-3,MAX_iter=100,echo=True):
        """
        Estimate parameters with the Baum-Welch algorithm.
        
        :param list seqs: Required: A list of mutated sequences.
          
        :param list param_to_estimate: A list of parameters to estimate. Options are:
          delta, tau, epsilon, gamma. The default are ["delta","tau"].
          
        :param str method: The estimating method to use. Options are: MLE and MAP.
          The default is MLE.
        
        :param dictionary lower_bound: The lower bound of parameters. The format is the
          same as upper bound. By default, the lower bound is set to 0.20 for all 
          parameters.
        
        :param float tol: stop the estimation if the percentage of likelihood increase is less 
          than the input value of tol.
        
        :param int MAX_iter: the maximum number of iterations to perform.
        
        :param boolean echo: Print the likelihood at each iteration.
        """
        
        
        seqs = [s for s in seqs if s != self.reference_sequence]
        self.param_to_estimate = param_to_estimate
        if method == "Unconstrained_MLE":
            pass
        elif method == "Constrained_MLE":
            if not lower_bound:
                self.lower_bound = {"delta": [0.1 for _ in range(self.n)] + [0],
                                    "tau": [0.1 for _ in range(self.n+1)],
                                    "MM": [0.2 for _ in range(self.n+1)],
                                    "epsilon": [0] + [0.5 for _ in range(self.n-1)] + [0],
                                    "DM": [0.1 for _ in range(self.n+1)],
                                    "gamma": [0.5 for _ in range(self.n+1)],
                                    "IM": [0.1 for _ in range(self.n+1)]}
            else:
                self.lower_bound = lower_bound
            self._check_bound()
        else:
            raise ValueError("Method need to be set to either Unconstrained_MLE or Constrained_MLE.")
        
        for i in range(MAX_iter+1):
            if echo and i != 0:
                print("Iteration %d, model likelihood %.2e" %(i, self.likelihood))

            param_hat, likelihood = self.EM(seqs,ncores=ncores)
            
            if abs(likelihood-self.likelihood) / self.likelihood < tol:
                break
            
            self.update_parameters(method, param_hat)
            self.likelihood = likelihood
        if echo:
            print("Complete at iteration %d, model likelihood %.2e" %(i, self.likelihood))
    

    
    
    def forward(self,t):
        """
        Compute matrices with the forward algorithm.
        
        :param str t: A mutated sequence.
        
        :return: Three matrices computed by the forward algorithm: (match, deletion,
          insertion).
        :rtype: tuple
        """
      
        dim_i = self.n
        dim_j = len(t)

        AM = [[self._init_AM(i,j) for j in range(dim_j + 1)] for i in range(dim_i + 1)]
        AD = [[self._init_AD(i,j) for j in range(dim_j + 1)] for i in range(dim_i + 1)]
        AI = [[self._init_AI(i,j) for j in range(dim_j + 1)] for i in range(dim_i + 1)]

        for i in range(1, dim_i + 1):
            for j in range(1, dim_j + 1):
                M_candidates = [(1-self.delta[i-1]-self.tau[i-1]) * AM[i-1][j-1],  \
                                    (1-self.epsilon[i-1]-self.rho) * AD[i-1][j-1], \
                                    (1-self.gamma[i-1]-self.pi) * AI[i-1][j-1]]
                D_candidates = [self.delta[i-1] * AM[i-1][j], \
                                self.epsilon[i-1] * AD[i-1][j],  \
                                self.pi * AI[i-1][j]]
                I_candidates = [self.tau[i] * AM[i][j-1], \
                                self.rho * AD[i][j-1],      \
                                self.gamma[i] * AI[i][j-1]]
                AM[i][j] = self._match(t, i-1, j-1) * sum(M_candidates)
                AD[i][j] = self.q * sum(D_candidates)
                AI[i][j] = self.q * sum(I_candidates)
        return np.array(AM), np.array(AD), np.array(AI)




    def backward(self, t):
        """
        Compute matrices with the backward algorithm.
        
        :param str t: A mutated sequence.
        
        :return: Three matrices computed by the backward algorithm: (match, deletion,
          insertion).
        :rtype: tuple
        """
      
      
        dim_i = self.n + 1
        dim_j = len(t) + 1

        BM = [[self._init_B(i,j,t,"M") for j in range(dim_j + 1)] for i in range(dim_i + 1)]
        BD = [[self._init_B(i,j,t,"D") for j in range(dim_j + 1)] for i in range(dim_i + 1)]
        BI = [[self._init_B(i,j,t,"I") for j in range(dim_j + 1)] for i in range(dim_i + 1)]

        for ii in range(1, dim_i+1):
            for jj in range(1, dim_j+1):
                i = dim_i - ii
                j = dim_j - jj

                if i == dim_i-1 and j == dim_j-1:
                    continue        
                M_candidates = [(1-self.delta[i]-self.tau[i]) * self._match(t, i, j) * BM[i+1][j+1], \
                                self.delta[i] * self.q * BD[i+1][j], \
                                self.tau[i] * self.q * BI[i][j+1]]
                D_candidates = [(1-self.epsilon[i]-self.rho) * self._match(t, i, j) * BM[i+1][j+1], \
                                self.epsilon[i] * self.q * BD[i+1][j], \
                                self.rho * self.q * BI[i][j+1]]
                I_candidates = [(1-self.gamma[i]-self.pi) * self._match(t, i, j) * BM[i+1][j+1], \
                                self.pi * self.q * BD[i+1][j], \
                                self.gamma[i] * self.q * BI[i][j+1]]

                BM[i][j] = sum(M_candidates) 
                BD[i][j] = sum(D_candidates)
                BI[i][j] = sum(I_candidates)
        return np.array(BM)[:-1,:-1], np.array(BD)[:-1,:-1], np.array(BI)[:-1,:-1]




    def calc_psi(self,state,AM,AD,AI,BM,BD,BI):
        """
        Calculate psi using the forward-backward matrices.
        
        :param str state: To compute the psi of which state. Options are:
          M (for match), D (for deletion), I (for insertion).
        
        :return: A matrix of psi.
        :rtype: numpy array
        """
        
        if state == "M":
            return np.sum(AM * BM, axis=1)
        if state == "D":
            return np.sum(AD * BD, axis=1)
        if state == "I":
            return np.sum(AI * BI, axis=1)
    
    
    
    
    def calc_xi(self,t,state1,state2,AM,AD,AI,BM,BD,BI):
        """
        Calculate xi (the marginal transition probability from state1 -> state2) using the 
        forward-backward matrices.
        
        :param str state1: The first state. Options are:
          M (for match), D (for deletion), I (for insertion).
          
        :param str state2: The second state. Options are:
          M (for match), D (for deletion), I (for insertion).
        
        :return: A matrix of psi.
        :rtype: numpy.ndarray
        """
      
        
        dim_i = self.n
        dim_j = len(t)
        xi = []
        for i in range(dim_i+1):
            xi_tmp = []
            for j in range(dim_j+1):
                if i < dim_i:
                    if state1 == "M" and state2 == "D":
                        xi_tmp.append(AM[i][j] * self.delta[i] * self.q * BD[i+1][j])
                    if state1 == "D" and state2 == "D":
                        xi_tmp.append(AD[i][j] * self.epsilon[i] * self.q * BD[i+1][j])
                    if state1 == "I" and state2 == "D":
                        xi_tmp.append(AI[i][j] * self.pi * self.q * BD[i+1][j])    
                    
                if j < dim_j:
                    if state1 == "M" and state2 == "I":
                        xi_tmp.append(AM[i][j] * self.tau[i] * self.q * BI[i][j+1])
                    if state1 == "I" and state2 == "I":
                        xi_tmp.append(AI[i][j] * self.gamma[i] * self.q * BI[i][j+1])
                    if state1 == "D" and state2 == "I":
                        xi_tmp.append(AD[i][j] * self.rho * self.q * BI[i][j+1])
                        
                if i < dim_i and j < dim_j:
                    if state1 == "D" and state2 == "M":
                        xi_tmp.append(AD[i][j] * (1-self.epsilon[i]-self.rho) * self._match(t,i,j) * BM[i+1][j+1])
                    if state1 == "I" and state2 == "M":
                        xi_tmp.append(AI[i][j] * (1-self.gamma[i]-self.pi) * self._match(t,i,j) * BM[i+1][j+1])
                    if state1 == "M" and state2 == "M":
                        xi_tmp.append(AM[i][j] * (1-self.delta[i]-self.tau[i]) * self._match(t,i,j) * BM[i+1][j+1])       
            xi.append(sum(xi_tmp))
        if state1 == "I" and state2 == "M":
            xi[-1] = AI[-1][-1] * (1 - self.gamma[self.n])
        if state1 == "M" and state2 == "M":
            xi[-1] = AM[-1][-1] * (1 - self.tau[self.n])            
        return np.array(xi)
    
    
    
    
    def EM(self,seqs,ncores=1):
        """
        Compute the updated parameters with the Baum-Welch's expectation maximization procedure.
        
        :param list seqs: Required: A list of mutated sequences.
          
        :param str method: The estimating method to use. Options are: MLE and MAP.
          The default is MLE.
        
        :return: The updated parameters in a dictionary and the likelihood of the model.
          
        :rtype: tuple
        """
      
      
        param_hat = {}
        for param in self.param_to_estimate:
            param_hat[param] = np.zeros((self.n+1,2))
        
        seqs_and_count = [(seq,seqs.count(seq)) for seq in set(seqs)]
        
        if ncores==1:
            list_of_param_hat = []
            for t,c in seqs_and_count:
                list_of_param_hat.append(self.compute_param_hat(t))
        elif ncores > 1:
            pool = mp.Pool(ncores)
            with pool:
                list_of_param_hat = pool.map(self.compute_param_hat, [s[0] for s in seqs_and_count])
        
        for i in range(len(seqs_and_count)):
            t,c = seqs_and_count[i]
            param_hat_one_sequence = list_of_param_hat[i]
            for param in param_hat_one_sequence:
                param_hat[param][:,0] += param_hat_one_sequence[param]["numerator"] * c
                param_hat[param][:,1] += param_hat_one_sequence[param]["denominator"] * c
        
        if "delta" in param_hat:
            likelihood = param_hat["delta"][0,1]
        elif "tau" in param_hat:
            likelihood = param_hat["tau"][0,1]
        else:
            likelihood = -float("inf")
        
        for param in param_hat:
            tmp = param_hat[param]
            param_hat[param] = [k[0]/k[1] if k[1] != 0 else 0 for k in tmp]            
        return param_hat, likelihood

    
    
    
    def compute_param_hat(self,t):
        AM,AD,AI = self.forward(t)
        BM,BD,BI = self.backward(t)
        param_hat = {}
        param_to_state = {"delta":["M","D"],
                          "tau":["M","I"],
                          "epsilon":["D","D"],
                          "rho":["D","I"],
                          "gamma":["I","I"],
                          "pi":["I","D"]}
        
        for param_group in [["delta","tau"],["epsilon","rho"],["gamma","pi"]]:
            p = list(set(param_group).intersection(set(self.param_to_estimate)))
            if len(p) == 2:
                states = param_to_state[p[0]]
                denominator = self.calc_psi(states[0],AM,AD,AI,BM,BD,BI)
                denominator = 0.5 * (denominator + max(denominator))
                param_hat[p[0]] = {"numerator": self.calc_xi(t,states[0],\
                                       states[1],AM,AD,AI,BM,BD,BI),
                                   "denominator":denominator}
                states = param_to_state[p[1]]
                param_hat[p[1]] = {"numerator": self.calc_xi(t,states[0],\
                                       states[1],AM,AD,AI,BM,BD,BI),
                                   "denominator":denominator}

            elif len(p) == 1:
                states = param_to_state[p[0]]
                p_diff = list(set(param_group) - set(self.param_to_estimate))[0]
                numerator = self.calc_xi(t,states[0],states[1],AM,AD,AI,BM,BD,BI)
                denominator = numerator + \
                            self.calc_xi(t,states[0],"M",AM,AD,AI,BM,BD,BI)
                denominator = 0.5 * (denominator + max(denominator))
                param_hat[p[0]] = {"numerator": (1 - np.array(\
                                        getattr(self, p_diff))) * numerator,
                                   "denominator":denominator}
        return param_hat
    
    
    
    
    def update_parameters_adaptive_step_size(self, param, param_hat, lb):
        if "delta" in param_hat and "tau" in param_hat:
            p = "MM"
            last_value = 1 - param["delta"] - param["tau"]
            last_value_hat = 1 - param_hat["delta"] - param_hat["tau"]
        elif len(param_hat) == 1:
            if "epsilon" in param_hat:
                p = "DM"
                last_value = 1 - self.rho - param["epsilon"]
                last_value_hat = 1 - self.rho - param_hat["epsilon"]
            elif "gamma" in param_hat:
                p = "IM"
                last_value = 1 - self.pi - param["gamma"]
                last_value_hat = 1 - self.pi - param_hat["gamma"]
            elif "delta" in param_hat:
                p = "MM"
                last_value = 1 - param["tau"] - param["delta"]
                last_value_hat = 1 - param["tau"] - param_hat["delta"]
                param.pop("tau")
            elif "tau" in param_hat:
                p = "MM"
                last_value = 1 - param["delta"] - param["tau"]
                last_value_hat = 1 - param["delta"] - param_hat["tau"]
                param.pop("delta")
        param[p] = last_value
        param_hat[p] = last_value_hat

        param[p] = last_value
        vector = dict([(p, param_hat[p]-param[p]) for p in param_hat])
        for v in vector:
            if abs(vector[v]-0) < 1e-4:
                vector[v] = -1
        
        step_size = [(lb[p]-param[p])/vector[p] for p in param]
        step_size = [s for s in step_size if s >= 0]
        if len(step_size) > 0:
            step_size = min(step_size + [1]) * 0.999
        else:
            step_size = 0
        result = {}
        for p in param_hat:
            result[p] = param[p] + step_size * vector[p]
        return result         
        
    
    
    
    def update_parameters(self, method, param_hat):
        if method == "Unconstrained_MLE":
            for param in param_hat:
                setattr(self, param, param_hat[param])
        
        elif method == "Constrained_MLE":
            result = {}
            if "delta" in param_hat and "tau" in param_hat:
                result["delta"] = []
                result["tau"] = []
                for i in range(self.n+1):
                    param = {"delta":self.delta[i],\
                             "tau":self.tau[i]}
                    p_hat = {"delta":param_hat["delta"][i],\
                             "tau":param_hat["tau"][i]}
                    lb = {"delta":self.lower_bound["delta"][i],\
                          "tau":self.lower_bound["tau"][i],
                          "MM":self.lower_bound["MM"][i]}
                    
                    tmp = self.update_parameters_adaptive_step_size(param, p_hat, lb)
                    result["delta"].append(tmp["delta"])
                    result["tau"].append(tmp["tau"])
                param_hat.pop("delta")
                param_hat.pop("tau")
            
            p1 = None
            for p in param_hat:
                result[p] = []
                if p == "delta":
                    p1 = "tau"
                    p2 = "MM"
                elif p == "tau":
                    p1 = "delta"
                    p2 = "MM"
                elif p == "epsilon":
                    p2 = "DM"
                elif p == "gamma":
                    p2 = "IM"
                for i in range(self.n+1):
                    param = {p:getattr(self, p)[i]}
                    if p1:
                        param[p1] = getattr(self, p1)[i]
                    p_hat = {p:param_hat[p][i]}
                    lb = {p:self.lower_bound[p][i],
                          p2:self.lower_bound[p2][i]}
                    tmp = self.update_parameters_adaptive_step_size(param, p_hat, lb)
                    result[p].append(tmp[p])
                p1 = None
            for param in result:
                setattr(self, param, result[param])
 
    
    
    
    def _check_parameters(self):
        all_probs = self.delta + self.tau + self.epsilon + self.gamma + [self.rho,self.q,self.match]
        if not all([p>=0 and p<=1 for p in all_probs]):
            raise ValueError("Probability needs to be in range [0,1].")
        
        if len(self.delta) != self.n + 1 or self.delta[-1] != 0:
            raise ValueError("Delta needs to be a list of %d values. The last value needs to be a placeholder 0."\
                             % (self.n + 1))
            
        if len(self.tau) != self.n + 1:
            raise ValueError("Tau needs to be a list of %d values." % (self.n + 1))
            
        if len(self.epsilon) != self.n + 1 or self.epsilon[0] != 0 or self.epsilon[-1] != 0:
            raise ValueError("Epsilon needs to be a list of %d values. The first and the last values need to be a placeholder 0."\
                            % (self.n + 1))
            
        if len(self.gamma) != self.n + 1:
            raise ValueError("Gamma needs to be a list of %d values." % (self.n + 1))
    
    
    def _check_bound(self):
        for param in self.param_to_estimate:
            if len(self.lower_bound[param]) != self.n + 1:
                raise ValueError("%s needs to be a list of %d values." % (param, self.n + 1))
            if any([i>=1 for i in self.lower_bound[param]]):
                raise ValueError("the values in the lower bound for %s need to less than 1." %param)
    
    
    def _log(self,value):
        if value == 0:
            return -float("inf")
        else:
            return log(value)
    
    
    def _match(self, t, i, j, use_log=False):
        if (i>=0 and i < self.n) and (j>=0 and j < len(t)):
            if self.reference_sequence[i] == t[j]:
                if use_log:
                    return self._log(self.match)
                else:
                    return self.match
            else:
                if use_log:
                    return self._log(self.mismatch)
                else:
                    return self.mismatch
        return 0
    
    
    def _init_bound(self, ):
        return lower_bound, upper_bound
    
    
    def _init_M(self, i, j):
        if j == 0 and i == 0:
            return 0
        return -float("inf")
    
    
    def _init_D(self, i, j):
        if i != 0 and j == 0:
            return self._log(self.delta[0] * self.q) + sum([self._log(self.epsilon[k] * self.q) for k in range(1,i)])
        return -float("inf")
    
    
    def _init_I(self, i, j):
        if i == 0 and j != 0:
            return self._log(self.tau[0] * self.q) + self._log(self.gamma[0] * self.q) * (j-1)
        return -float("inf")
    
    
    def _init_AM(self, i, j):
        if j == 0 and i == 0:
            return 1
        return 0


    def _init_AD(self, i, j):
        if i != 0 and j == 0:
            return self.delta[0] * self.q * np.prod([self.q * self.epsilon[k] for k in range(1,i)])
        return 0


    def _init_AI(self, i, j):
        if i == 0 and j != 0:
            return self.tau[0] * self.q * ((self.q * self.gamma[0]) ** (j-1))
        return 0


    def _init_B(self, i, j, t, state):
        dim_i = self.n + 1
        dim_j = len(t) + 1
        if i == dim_i-1 and j == dim_j-1:
            if state == "M":
                return 1 - self.tau[dim_i - 1]
            elif state == "D":
                return 1
            elif state == "I":
                return 1 - self.gamma[self.n]
        else:
            return 0
    
    
    def _init_R(self, i, j):
        if j == 0 and i == 0:
            return ["0","0","0"]
        elif i == 0:
            return ["0","D","I"]
        elif j == 0:
            return ["0","0","I"]
        else:
            return ["0","0","0"]
        
        
    def _viterbi_slow(self, t):
        """
        Perform the Viterbi algorithm to generate the best alignment between the
        reference sequence and a mutated sequence t.
        
        :param str t: A mutated sequence.
        
        :return: The alignment outcome: (reference sequence, mutated sequence, 
        alignment log(probability).)
        :rtype: tuple
        """
      
        M,D,I,R = self._viterbi_matrices(t)
        aligned_ref, aligned_t, BEST_SCORE = self._backtrace(t, M, D, I, R)
        return aligned_ref, aligned_t, BEST_SCORE
        
        
    def _viterbi_matrices(self, t):
        """
        Compute the matrices with the Viterbi's dynamic programming algorithm.
        
        :param str t: A mutated sequence.
        
        :return: Four matrices computed by the Viterbi algorithm: (match, deletion,
          insertion, back trace matrix).
        :rtype: tuple
        """
      
      
        MIN = -float("inf")
        dim_i = self.n + 1
        dim_j = len(t) + 1
        M = [[self._init_M(i, j) for j in range(0, dim_j)] for i in range(0, dim_i)]
        D = [[self._init_D(i, j) for j in range(0, dim_j)] for i in range(0, dim_i)]
        I = [[self._init_I(i, j) for j in range(0, dim_j)] for i in range(0, dim_i)]
        R = [[self._init_R(i, j) for j in range(0, dim_j)] for i in range(0, dim_i)]

        for i in range(1, dim_i):
            for j in range(1, dim_j):
                M_candidates = [self._log(1-self.delta[i-1]-self.tau[i-1]) + M[i-1][j-1], \
                                    self._log(1-self.epsilon[i-1]-self.rho) + D[i-1][j-1], \
                                    self._log(1-self.gamma[i-1]-self.pi) + I[i-1][j-1]]
                D_candidates = [self._log(self.delta[i-1]) + M[i-1][j], \
                                self._log(self.epsilon[i-1]) + D[i-1][j], \
                                self._log(self.pi) + I[i-1][j]]
                I_candidates = [self._log(self.tau[i]) + M[i][j-1], \
                                self._log(self.rho) + D[i][j-1], \
                                self._log(self.gamma[i]) + I[i][j-1]]
                
                M[i][j] = self._match(t, i-1, j-1, use_log=True) + \
                            max(M_candidates)
                D[i][j] = self._log(self.q) + max(D_candidates)
                I[i][j] = self._log(self.q) + max(I_candidates)

                if max(M_candidates) != MIN:
                    R[i][j][0] = ["M","D","I"][M_candidates.index(max(M_candidates))]
                if max(D_candidates) != MIN:   
                    R[i][j][1] = ["M","D","I"][D_candidates.index(max(D_candidates))]
                if max(I_candidates) != MIN:   
                    R[i][j][2] = ["M","D","I"][I_candidates.index(max(I_candidates))]
        return M,D,I,R
    
    
    def _backtrace(self, t, M, D, I, R):
        """
        Get the best alignment by backtrace through the matrices computed by the
        function viterbi_matrices(t).
        
        :param str t: A mutated sequence.
        
        :param M: the match matrix computed by viterbi_matrices.
        
        :param D: the deletion matrix computed by viterbi_matrices.
        
        :param I: the insertion matrix computed by viterbi_matrices.
        
        :param R: the back trace matrix computed by viterbi_matrices.
        
        :return: The alignment outcome: (reference sequence, mutated sequence, 
        alignment log(probability).)
        :rtype: tuple
        """
      
      
        R[1][1][0] = "0"
        R[0][1][1] = "0"
        R[0][1][2] = "0"
        R[1][0][2] = "0"
        scores = [M[self.n][len(t)] + \
                      self._log(1-self.tau[self.n]),\
                  D[self.n][len(t)],\
                  I[self.n][len(t)] + \
                      self._log(1-self.gamma[self.n])]
        BEST_SCORE = max(scores)
        state = ["M","D","I"][scores.index(BEST_SCORE)]

        aligned_ref = ''
        aligned_t = ''
        i = self.n
        j = len(t)
        while (state != "0"):
            if state == "M":
                state = R[i][j][0]
                aligned_ref += self.reference_sequence[i-1]
                aligned_t += t[j-1]
                i -= 1
                j -= 1

            elif state == "D":
                state = R[i][j][1]
                aligned_ref += self.reference_sequence[i-1]
                aligned_t += "-"
                i -= 1

            elif state == "I":
                state = R[i][j][2]
                aligned_ref += "-"
                aligned_t += t[j-1]
                j -= 1

        aligned_ref = ''.join([aligned_ref[j] for j in range(-1, -(len(aligned_ref)+1), -1)])
        aligned_t = ''.join([aligned_t[j] for j in range(-1, -(len(aligned_t)+1), -1)])
        return aligned_ref, aligned_t, BEST_SCORE
