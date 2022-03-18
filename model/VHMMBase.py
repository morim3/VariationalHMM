import jax.numpy as jnp
import jax.scipy.special import logsumexp


class VHMMBase:
    def __init__(self, cluster_num, initial_params):
        self.cluster_num = cluster_num
        self.params = initial_params.copy()
        pass

    def fit(self, X):
        pass

    def predict(self, X, viterbi=False):

        if not viterbi:
            return self.e_step(X)

        else:
            return self.viterbi(X)
        pass

    @staticmethod
    def _forward_e_step(prev_log_prob, trans_log_prob, obs_log_prob, obs_index):
        '''
        params:
        prev_log_prob: jnp array, shape (batch_size, hidden_num)
        trans_log_prob: jnp array, shape (hidden_num, hidden_num)
        obs_log_prob: jnp array, shape (batch_size, hidden_num)
        '''
        log_prob = jnp.expand_dims(prev_log_prob, axis=2) + trans_log_prob \
            + jnp.expand_dim(obs_log_prob, axis=2)
        return logsumexp(log_prob, axis=1)

    @staticmethod
    def _backward_e_step(next_log_prob, trans_log_prob, next_obs_log_prob):
        log_prob = jnp.expand_dims(next_log_prob, axis=2) + trans_log_prob + \ 
            jnp.expand_dim(obs_log_prob, axis=1)
        return logsumexp(log_prob, axis=2)


    @staticmethod
    def e_step(X):
        

        pass

    @staticmethod
    def viterbi(X):
        pass
  
