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
    with open("group_51.txt","w") as f:

        # Cookie problem
        hypos = ["Bowl1", "Bowl2"]
        priors = [0.5, 0.5]
        obs = ["chocolate", "vanilla"]
        likelihood = [[15/50, 35/50], [30/50, 20/50]]

        b = Bayes(hypos, priors, obs, likelihood)

        l = b.likelihood("chocolate", "Bowl1")
        n_c = b.norm_constant("vanilla")
        p_1 = b.single_posterior_update("vanilla", [0.5, 0.5])
        p_2 = b.compute_posterior(["chocolate", "vanilla"])

        f.write("Question 1 - Probability of vanilla cookie coming from Bowl 1: {}".format(round(p_1[0],3)))
        f.write("\nQuestion 2 - Probability of chocolate and vanilla cookie coming from Bowl 2: {}".format(round(p_2[1],3)))

        # Archer problem
        hypos = ["beginner", "intermediate", "advanced", "expert"]
        priors = [0.25, 0.25, 0.25, 0.25]
        obs = ["yellow", "red", "blue", "black", "white"]
        likelihood = [[0.05,0.1,0.4,0.25,0.2], [0.1,0.2,0.4,0.2,0.1], [0.2,0.4,0.25,0.1,0.05], [0.3,0.5,0.125,0.05,0.025]]

        b = Bayes(hypos, priors, obs, likelihood)
        
        p_3 = b.compute_posterior(["yellow", "white", "blue", "red", "red", "blue"])

        f.write("\nQuestion 3 - Probability of archer being intermediate: {}".format(round(p_3[1],3)))
        f.write("\nQuestion 4 - Most probable level of the archer: {}".format(hypos[p_3.index(max(p_3))]))
