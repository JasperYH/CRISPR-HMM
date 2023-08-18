def test_viterbi(mode,t):
    M,D,I,R = model.viterbi_matrices(t)
    aligned_ref, aligned_seq, best_score = model.backtrace(t, M, D, I, R)
    
    aligned_ref_by_iteration,aligned_seq_by_iteration,best_score_by_iteration = best_aln_by_iteration(model,t)
    assert(aligned_ref == alned_ref_by_iteration and 
           aligned_seq == aligned_seq_by_iteration and 
           best_score == best_score_by_iteration)


def test_EM(mode,t):
    pass


if __name__ == "__main__":
    s = "ATG"
    t = "ACG"

    delta = [0.2,0.4,0.3,0]
    tau = [0.2,0.25,0.2,0.2]
    epsilon = [0,0.2,0.3,0]
    gamma = 0.4
    rho = 0.05
    q = 1
    match = .9
    model = HMM_Model(reference_sequence=s,delta=delta, tau=tau, \
                      epsilon=epsilon, gamma=gamma, rho=rho, q=q, match=match)
    test_viterbi(model,t)
    
    