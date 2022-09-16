class Bayes:
    
    def __init__(self, hypos, priors, obs, likelihoods):
        self.hypos = hypos
        self.priors = priors
        self.obs = obs
        self.likelihoods = likelihoods
    
    def likelihood(self, observation, hypothesis):
        obs_index = self.obs.index(observation)
        hyp_index = self.hypos.index(hypothesis)
        return self.likelihoods[hyp_index][obs_index]

    def norm_constant(self, observation):
        obs_index = self.obs.index(observation)
        result = 0
        for i,hyp_likelihoods in enumerate(self.likelihoods):
            result += hyp_likelihoods[obs_index]*self.priors[i]
        return result
    
    def single_posterior_update(self, observation, priors):
        result = []
        for i, prior in enumerate(priors):
            result.append(prior*self.likelihood(observation,self.hypos[i])/self.norm_constant(observation))
        return result
    
    def compute_posterior(self, observations):
        posterior = self.priors
        for observation in observations:
            posterior = self.single_posterior_update(observation, posterior)
            self.priors = posterior
        return posterior
        


if __name__ == '__main__':
    hypos = ["Bowl1", "Bowl2"]
    priors = [0.5, 0.5]
    obs = ["chocolate", "vanilla"]
    # e.g. likelihood[0][1] corresponds to the likehood of Bowl1 and vanilla, or 35/50
    likelihood = [[15/50, 35/50], [30/50, 20/50]]

    b = Bayes(hypos, priors, obs, likelihood)

    l = b.likelihood("chocolate", "Bowl1")
    print("likelihood(chocolate, Bowl1) = %s " % l)
    n_c = b.norm_constant("vanilla")
    print("normalizing constant for vanilla: %s" % n_c)
    p_1 = b.single_posterior_update("vanilla", [0.5, 0.5])
    print("vanilla - posterior: %s" % p_1)
    p_2 = b.compute_posterior(["chocolate", "vanilla"])
    print("chocolate, vanilla - posterior: %s" % p_2)


    hypos = ["beginner", "intermediate", "advanced", "expert"]
    priors = [0.25, 0.25, 0.25, 0.25]
    obs = ["yellow", "red", "blue", "black", "white"]
    likelihood = [[0.05,0.1,0.4,0.25,0.2], [0.1,0.2,0.4,0.2,0.1], [0.2,0.4,0.25,0.1,0.05], [0.3,0.5,0.125,0.05,0.025]]
    b = Bayes(hypos, priors, obs, likelihood)
    p_2 = b.compute_posterior(["yellow", "white", "blue", "red", "red", "blue"])
    print("archer - posterior: %s" % p_2)
