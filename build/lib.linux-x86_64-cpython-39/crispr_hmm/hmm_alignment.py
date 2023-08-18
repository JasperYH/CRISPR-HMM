import numpy as np
from math import log, e
import pathos.multiprocessing as mp
from functools import partial
import copy
from .fast_viterbi import fast_viterbi



class hmm_model(object):
    """
    A class to represent a Hidden Markov Model (HMM) for sequence alignment.
    
    Attributes:
    - reference_sequence (str): The sequence used as a reference for alignments.
    - delta (list of float): Transition probabilities from match state to deletion state.
    - tau (list of float): Transition probabilities from match state to insertion state.
    - epsilon (list of float): Transition probabilities from deletion state to deletion state.
    - gamma (list of float): Transition probabilities from insertion state to insertion state.
    - rho (float): Transition probability from deletion state to insertion state.
    - pi (float): Transition probability from insertion state to deletion state.
    - match (float): Emission probability for a match. 
    - mismatch (float): Emission probability for a mismatch, calculated as 1 - match.
    - q (float): Emission probability for both insertions and deletions. 

    Note:
    - The mismatch emission probability is automatically determined as 1 minus the match probability.
    - It is recommended that `q` be smaller than the match emission probability.
    
    Parameters:
    :param reference_sequence (str): The sequence to be used as a reference for alignments.
    :param delta (float or list of float, optional): Transition probabilities from match to deletion state. Defaults to 0.02.
    :param tau (float or list of float, optional): Transition probabilities from match to insertion state. Defaults to 0.02.
    :param epsilon (float or list of float, optional): Transition probabilities from deletion to deletion state. Defaults to 0.8.
    :param gamma (float or list of float, optional): Transition probabilities from insertion to insertion state. Defaults to 0.8.
    :param rho (float, optional): Transition probability from deletion to insertion state. Defaults to 0.01.
    :param pi (float, optional): Transition probability from insertion to deletion state. Defaults to 0.01.
    :param p (float, optional): Emission probability of a match. Defaults to 0.9.
    :param q (float, optional): Emission probability for insertions and deletions. Defaults to 0.8. It's recommended to be less than `p`.
    """
            
  
    def __init__(self, reference_sequence, delta=0.02, tau=0.02, epsilon=0.8,
                 gamma=0.8, rho=0.01, pi=0.01, p=0.9, q=0.8):
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
    
    
    
    
    def calculate_alignment_probability(self, aligned_ref, aligned_seq, fix_trivial_param=True, use_log=False):
        """
        Calculate the probability of a given alignment based on a Hidden Markov Model (HMM).

        Parameters:
        -----------
        aligned_ref : str
            The aligned reference sequence with "-" denoting a gap.

        aligned_seq : str
            The aligned mutated or query sequence with "-" denoting a gap.

        fix_trivial_param : bool, optional (default=True)
            If set to True, prohibits certain transition probabilities from changing from site to site.

        use_log : bool, optional (default=False)
            If set to True, returns the logarithm of the probability.

        Returns:
        --------
        prob : float
            The probability (or its logarithm) of the alignment.

        Notes:
        ------
        The alignment probability is calculated by multiplying emission probabilities 
        (match, mismatch, or gap) with the respective transition probabilities for each position.
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
        """
        Computes the best sequence alignment using the Viterbi algorithm.

        Parameters:
        -----------
        seqs : list of str
            A list of sequences to be aligned.

        fix_trivial_param : bool, optional (default=True)
            If set to True, this prohibits the transition probabilities from M -> M, D -> M, 
            and I -> M to change from site to site. 

        ncores : int, optional (default=1)
            The number of cores to use for parallel processing. If set to 1, no parallel 
            processing will be used.

        Yields:
        -------
        alignment : tuple
            An alignment result for each sequence in the list of sequences. The exact nature 
            of the alignment result depends on the underlying implementation of the 
            `fast_viterbi` function.

        Notes:
        ------
        The Viterbi algorithm is a dynamic programming algorithm that is used to find the 
        most likely sequence of states that would produce a given sequence of observations.
        """
        
        
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
    
    
    
       
    def forward(self,t):
        """
        Computes the forward probability matrices for a given sequence using 
        the forward algorithm.

        Parameters:
        -----------
        t : str
            The input sequence for which forward matrices are computed.

        Returns:
        --------
        AM, AD, AI : tuple of numpy.ndarray
            Three matrices determined by the forward algorithm, corresponding to 
            the forward probabilities for match, deletion, and insertion states,
            respectively.

        Notes:
        ------
        The forward algorithm is crucial for determining the likelihood of 
        a partial sequence up to a given position, given the current state. 
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
        Computes the backward probability matrices for a given sequence using 
        the backward algorithm.

        Parameters:
        -----------
        t : str
            The input sequence for which backward matrices are computed.

        Returns:
        --------
        BM, BD, BI : tuple of numpy.ndarray
            Three matrices determined by the backward algorithm, corresponding to 
            the backward probabilities for match, deletion, and insertion states,
            respectively.

        Notes:
        ------
        The backward algorithm is essential for determining the likelihood of 
        the partial sequence from a given position to the end, given the current 
        state.
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
        Calculates the psi values for a given state using both the forward (A) 
        and backward (B) matrices. Psi values offer insights into the likelihood 
        of being in a specified state.

        Parameters:
        -----------
        state : str
            The state for which the psi values are computed. Options include:
            - M (for match)
            - D (for deletion)
            - I (for insertion)

        AM : numpy.ndarray
            The forward matrix for the match state.

        AD : numpy.ndarray
            The forward matrix for the deletion state.

        AI : numpy.ndarray
            The forward matrix for the insertion state.

        BM : numpy.ndarray
            The backward matrix for the match state.

        BD : numpy.ndarray
            The backward matrix for the deletion state.

        BI : numpy.ndarray
            The backward matrix for the insertion state.

        Returns:
        --------
        psi_values : numpy.ndarray
            An array containing the computed psi values for the specified state 
            over the length of the sequence.

        Notes:
        ------
        The function computes the psi values by element-wise multiplication of 
        the forward and backward matrices for the specified state and then 
        summing across columns.
        """
    
    
    
    
    def calc_xi(self,t,state1,state2,AM,AD,AI,BM,BD,BI):
        """
        Calculates the xi values representing the transition probabilities between 
        `state1` and `state2`, given a sequence and the model parameters. The xi 
        values provide insights into the likelihood of a transition from one state 
        to another throughout the observed sequence.

        Parameters:
        -----------
        t : str
            The observed sequence based on which the xi values are computed.

        state1 : str
            The originating state for the transition. Options include:
            - M (for match)
            - D (for deletion)
            - I (for insertion)

        state2 : str
            The destination state for the transition. Options include:
            - M (for match)
            - D (for deletion)
            - I (for insertion)

        AM : numpy.ndarray
            The forward matrix for the match state.

        AD : numpy.ndarray
            The forward matrix for the deletion state.

        AI : numpy.ndarray
            The forward matrix for the insertion state.

        BM : numpy.ndarray
            The backward matrix for the match state.

        BD : numpy.ndarray
            The backward matrix for the deletion state.

        BI : numpy.ndarray
            The backward matrix for the insertion state.

        Returns:
        --------
        xi_values : numpy.ndarray
            An array containing the computed xi values for the specified transition 
            over the length of the sequence.

        Notes:
        ------
        This function iteratively computes xi values for each position in the sequence.
        It takes into consideration both the forward and backward probabilities and 
        integrates them with the model parameters to obtain a measure of the likelihood 
        of the desired transition.
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
 
    
    
    
    def compute_param_hat(self, t, param_to_estimate=["delta","tau"]):
        """
        Computes the estimated parameters for a given sequence using the forward-backward algorithm. 

        Parameters:
        -----------
        t : str
            The observed sequence for which the parameter estimations are computed.

        param_to_estimate : list of str, optional
            The list of parameters to estimate. By default, it estimates "delta" (transition from match 
            to deletion) and "tau" (transition from match to insertion). Possible values in the list are:
            - "delta"   : Transition probability from match to deletion.
            - "tau"     : Transition probability from match to insertion.
            - "epsilon" : Transition probability from deletion to deletion.
            - "gamma"   : Transition probability from insertion to insertion.

        Returns:
        --------
        dict
            A dictionary where the keys are the names of the estimated parameters and the values are 
            their corresponding estimations. The estimated values are derived as the ratio of the expected 
            number of specific transitions to the total expected number of transitions originating from 
            the match state.
        """
        
        AM,AD,AI = self.forward(t)
        BM,BD,BI = self.backward(t)
        param_hat = {}
        param_to_state = {"delta":["M","D"],
                          "tau":["M","I"],
                          "epsilon":["D","D"],
                          "rho":["D","I"],
                          "gamma":["I","I"],
                          "pi":["I","D"]}
        
        denominator = self.calc_psi("M",AM,AD,AI,BM,BD,BI)[0]
        for p in param_to_estimate:
            states = param_to_state[p]
            param_hat[p] = self.calc_xi(t,states[0],\
                           states[1],AM,AD,AI,BM,BD,BI) / denominator
        return param_hat
        

        
        
    def estimate_param(self, seqs, min_bound=None, max_bound=None, 
                     param_to_estimate=["delta","tau"], ncores=1):
        """
        Estimates model parameters from a collection of sequences using the forward-backward 
        algorithm and then applies min-max transformation to ensure the estimated parameters 
        lie within specified bounds.

        Parameters:
        -----------
        seqs : list of str
            A list of observed sequences based on which the parameters are to be estimated.

        min_bound : dict, optional
            A dictionary specifying the lower bounds for the parameters to be estimated. If not 
            provided, the current values of the parameters multiplied by two are used as default.

        max_bound : dict, optional
            A dictionary specifying the upper bounds for the parameters to be estimated. If not 
            provided, the minimum between twice the current values of the parameters and 0.5 is used.

        param_to_estimate : list of str, optional
            The list of parameters to estimate. By default, it estimates "delta" (transition from match 
            to deletion) and "tau" (transition from match to insertion). 

        ncores : int, optional
            Number of cores to use for parallel processing. If set to 1, no parallel processing is done. 
            For values greater than 1, the provided number of cores will be used for parallel estimation.

        Returns:
        --------
        None
            This method updates the object's attributes in place.

        Notes:
        ------
        This function first calculates the estimates (param_hat) for each sequence. If there are repeated 
        sequences in the input, it efficiently computes their combined effect. After aggregation, 
        the estimated parameters are then transformed to ensure they lie within the provided bounds 
        (or default bounds if none are given) using a min-max transformation.
        """
            
        if not min_bound:
            min_bound = {}
            for param in param_to_estimate:
                min_bound[param] = getattr(self,param)[1]
        
        if not max_bound:
            max_bound = {}
            for param in param_to_estimate:
                max_bound[param] = min(2 * getattr(self,param)[1], 0.5)
        
        param_hat = {}
        for param in param_to_estimate:
            param_hat[param] = np.zeros(self.n+1)
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
                param_hat[param] += param_hat_one_sequence[param] * c / len(seqs)
                
        for param in param_hat:
            param_transformed = [min_bound[param]-0.001] + self._min_max_transformation(param_hat[param][1:-1],
                                      min_bound[param],max_bound[param]) + [min_bound[param]+0.001]
          
            setattr(self,param,param_transformed)

    

    
    def _min_max_transformation(self, data, a, b):
        min_value = min(data)
        max_value = max(data)

        normalized_data = [(x - min_value) / (max_value - min_value) * (b - a) + a for x in data]
        return normalized_data
        
        
    
    
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
