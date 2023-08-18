import numpy as np
from math import log, e

class HMM_Model:
    def __init__(self, reference_sequence, delta, tau, epsilon, gamma, rho=0.03, pi=0.03, q=1, match=0.9):
        self.reference_sequence = reference_sequence
        self.n = len(self.reference_sequence)
        self.delta = delta
        self.tau = tau
        self.epsilon = epsilon
        self.gamma = gamma
        self.rho = rho
        self.pi = pi
        self.q = q
        self.match = match
        self.mismatch = 1 - self.match
        self._check_parameters()
        self.likelihood = 1e-100
    
    def get_likelihood(self, aligned_ref, aligned_seq, use_log=False, use_end_prob=True):
        prob = 1
        hidden_state = "S"
        position = 0
        for i in range(len(aligned_ref)):
            deltaC = self.delta[position]
            tauC = self.tau[position]
            epsilonC = self.epsilon[position]
            gammaC = self.gamma[position]
            transition = {("S","M"):(1-self.tau[0]-self.delta[0]),("S","D"):self.delta[0],("S","I"):self.tau[0],
                          ("M","M"):(1-tauC-deltaC),("M","D"):deltaC,("M","I"):tauC,
                          ("D","M"):(1-epsilonC-self.rho),("D","D"):epsilonC,("D","I"):self.rho,
                          ("I","M"):(1-gammaC-self.pi),("I","I"):gammaC,("I","D"):self.pi}

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
        if use_end_prob:
            transition = {("M","E"):(1-self.tau[position]),("I","E"):1-self.gamma[position],("D","E"):1}
            prob *= transition[(hidden_state,"E")]
        if use_log:
            return self._log(prob)
        return prob
    
    def viterbi(self, t):
        M,D,I,R = self.viterbi_matrices(t)
        aligned_ref, aligned_t, BEST_SCORE = self.backtrace(t, M, D, I, R)
        return aligned_ref, aligned_t, BEST_SCORE
    
    def Baum_Welch_estimator(self,seqs,method="MLE",param_to_learn=["delta","tau"],\
                             pseudocount=None,upper_bound=None,lower_bound=None,echo=True,MAX_iter=100,tol=1e-2):
        if method == "MLE":
            pass
        elif method == "MAP":
            self._check_pseudocount(pseudocount)
        else:
            raise ValueError("Method need to be set to either MLE or MAP.") 
        
#         upper_bound, lower_bound = self._check_bound(param_to_learn, upper_bound, upper_bound)
        
        for i in range(MAX_iter+1):
            if echo and (i-1) % 1 == 0:
                print("Iteration %d, model likelihood %.2e" %(i, self.likelihood))
            param_hat, likelihood = self.update_parameters(seqs,method=method,\
                                         param_to_learn=param_to_learn,pseudocount=pseudocount)
            
            if abs(likelihood-self.likelihood) / self.likelihood < tol:
                break
            
            for param in param_hat:
                val = [min(param_hat[param][i],upper_bound[param][i]) for i in range(self.n + 1)]
                val2 = [max(val[i],lower_bound[param][i]) for i in range(self.n + 1)]
                if param == "match":
                    self.mismatch = 1 - self.match
                
                setattr(self, param, val2)
            
            self.likelihood = likelihood
        
        if (i-1) % 10 != 0:
            print("Complete at iteration %d, model likelihood %.2e" %(i, self.likelihood))
    
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
    
    def _check_pseudocount(self,pseudocount):
        all_counts = pseudocount["delta"] + pseudocount["tau"] + pseudocount["epsilon"] + \
                     [pseudocount["gamma"]]
        if not all([c>=0 and c<pseudocount["denominator"] for c in all_counts]):
            raise ValueError("Pseudocount delta, tau, epsilon, gamma needs to greater or equal to 0 and smaller than denominator.")
        
        if len(pseudocount["delta"]) != self.n + 1 or pseudocount["delta"][-1] != 0:
            raise ValueError("Delta needs to be a list of %d values. The last value needs to be a placeholder 0."\
                             % (self.n + 1))
            
        if len(pseudocount["tau"]) != self.n + 1:
            raise ValueError("Tau needs to be a list of %d values." % (self.n + 1))
            
        if len(pseudocount["epsilon"]) != self.n + 1 or pseudocount["epsilon"][0] != 0 or pseudocount["epsilon"][-1] != 0:
            raise ValueError("Epsilon needs to be a list of %d values. The first and the last values need to be a placeholder 0."\
                            % (self.n + 1))
    
    def _check_bound(self, param_to_learn, upper_bound, lower_bound):
        for param in param_to_learn:
            if len(upper_bound[param]) != self.n + 1:
                raise ValueError("%s needs to be a list of %d values." % (param, self.n + 1))
            sum_of_bound = [lower_bound[param][i]+upper_bound[param][i] for i in range(self.n+1)]
            if any([i>=1 for i in sum_of_bound]):
                raise ValueError("%the sum of lower and upper bound for %s need to less than 1.")
    
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
        
    def viterbi_matrices(self, t):
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
    
    def backtrace(self, t, M, D, I, R):    
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
        
    def forward(self,t):
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
        if state == "M":
            return np.sum(AM * BM, axis=1)
        if state == "D":
            return np.sum(AD * BD, axis=1)
        if state == "I":
            return np.sum(AI * BI, axis=1)
    
    def calc_xi(self,t,state1,state2,AM,AD,AI,BM,BD,BI):
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
                if j < dim_j:
                    if state1 == "M" and state2 == "I":
                        xi_tmp.append(AM[i][j] * self.tau[i] * self.q * BI[i][j+1])
                    if state1 == "I" and state2 == "I":
                        xi_tmp.append(AI[i][j] * self.gamma[i] * self.q * BI[i][j+1])
                if i < dim_i and j < dim_j:
                    if state1 == "D" and state2 == "M":
                        xi_tmp.append(AD[i][j] * (1-self.epsilon[i]-self.rho) * self._match(t,i,j) * BM[i+1][j+1])
                    if state1 == "I" and state2 == "M":
                        xi_tmp.append(AI[i][j] * (1-self.gamma[i]-self.pi) * self._match(t,i,j) * BM[i+1][j+1])
            xi.append(sum(xi_tmp))
        if state1 == "I" and state2 == "M":
            xi[-1] = AI[-1][-1] * (1 - self.gamma[self.n])
        return xi
    
    def update_parameters(self,seqs,method="MLE",pseudocount=None,param_to_learn=["delta","tau"]):        
        param_hat = {}
        for param in param_to_learn:
            if param == "match":
                param_hat["match"] = np.zeros((1,2))
            else:
                param_hat[param] = np.zeros((self.n+1,2))
        
        seqs_and_count = [(seq,seqs.count(seq)) for seq in set(seqs)]
        
        for t,c in seqs_and_count:
            AM,AD,AI = self.forward(t)
            BM,BD,BI = self.backward(t)
            
            for param in param_to_learn:
                if param == "delta":
                    numerator = self.calc_xi(t,"M","D",AM,AD,AI,BM,BD,BI)
                    denominator = self.calc_psi("M",AM,AD,AI,BM,BD,BI)
                if param == "tau":
                    numerator = self.calc_xi(t,"M","I",AM,AD,AI,BM,BD,BI)
                    denominator = self.calc_psi("M",AM,AD,AI,BM,BD,BI)
                if param == "epsilon":
                    numerator = self.calc_xi(t,"D","D",AM,AD,AI,BM,BD,BI)
                    denominator = self.calc_psi("D",AM,AD,AI,BM,BD,BI)
                if param == "gamma":
                    numerator = self.calc_xi(t,"I","I",AM,AD,AI,BM,BD,BI)
                    denominator = self.calc_psi("I",AM,AD,AI,BM,BD,BI)
                if param == "match":
                    match_matrix = np.array([[int(self.reference_sequence[i]==t[j]) \
                          for j in range(len(t))] for i in range(self.n)])
                    numerator = np.sum((AM * BM) [1:,1:] * match_matrix)
                    denominator = np.sum((AM * BM) [1:,1:])
                param_hat[param][:,0] += np.array(numerator) * c
                param_hat[param][:,1] += np.array(denominator) * c
        
        if "delta" in param_hat:
            likelihood = param_hat["delta"][0,1]
        else:
            likelihood = -float("inf")
        
        if method == "MLE":
            for param in param_hat:
                tmp = param_hat[param]
                param_hat[param] = [k[0]/k[1] if k[1] != 0 else 0 for k in tmp]
        elif method == "MAP":
            pass   
        return param_hat, likelihood
