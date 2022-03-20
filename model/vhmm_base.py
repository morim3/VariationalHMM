import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax import lax, jit


class HMMBase:
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

    @jit
    @staticmethod
    def _forward_one_step(alpha, trans_log_prob, obs_log_prob, ):
        '''
        :param alpha: jnp array, shape (batch_size, hidden_num)
        :param trans_log_prob: jnp array, shape (hidden_num, hidden_num)
        :param obs_log_prob: jnp array, shape (batch_size, hidden_num)
        :return:
        '''
        log_prob = jnp.expand_dims(alpha, axis=2) + jnp.expand_dims(trans_log_prob, axis=0) \
            + jnp.expand_dims(obs_log_prob, axis=2)
        result = logsumexp(log_prob, axis=1)
        cond_log_likelihood = logsumexp(result, axis=1)
        return result, cond_log_likelihood

    @staticmethod
    @jit
    def _backward_one_step(beta, trans_log_prob, obs_log_prob, cond_log_likelihood):
        '''

        :param next_log_prob:
        :param trans_log_prob:
        :param next_obs_log_prob:
        :return:
        '''
        log_prob = jnp.expand_dims(beta, axis=1) + jnp.expand_dims(trans_log_prob, axis=0) + \
            jnp.expand_dims(obs_log_prob, axis=1)

        result = logsumexp(log_prob, axis=2) - cond_log_likelihood

        return result

    @staticmethod
    @jit
    def forward_step(init_log_prob, obs_log_probs, trans_log_prob):

        def scan_fn(log_prob, obs_log_prob):
            result, partition = HMMBase._forward_one_step(log_prob, trans_log_prob, 
                obs_log_prob)
            normed_result = result - partition
            return (
                normed_result,
                [normed_result, partition]
            )

        _, [forward_prob, cond_log_likelihood] = lax.scan(scan_fn, init_log_prob, obs_log_probs)
        return forward_prob, cond_log_likelihood

    @staticmethod
    @jit
    def backward_step(obs_log_probs, trans_log_prob, cond_log_likelihoods):

        def scan_fn(beta, obs_log_prob, cond_log_likelihood):
            result = HMMBase._backward_one_step(beta, trans_log_prob,
                obs_log_prob, cond_log_likelihood)
            return (
                result,
                result
            )

        _, backward_prob = lax.scan(scan_fn, jnp.zeros_like(obs_log_probs[0]), [obs_log_probs, cond_log_likelihoods], reverse=True)
        return backward_prob

    @staticmethod
    @jit
    def _e_step(obs_log_probs, trans_log_prob, initial_log_prob):
        """
        :params: obs_log_probs: shape(time, batch, hidden)
        """

        initial_forward = obs_log_probs[0] + jnp.expand_dims(initial_log_prob, axis=(0, 1))
        initial_log_likelihood = logsumexp(initial_forward, axis=2)
        initial_forward = initial_forward - initial_log_likelihood

        forward_prob, cond_log_likelihoods = HMMBase.forward_step(initial_forward, obs_log_probs[1:], trans_log_prob)
        forward_prob = jnp.concatenate([initial_forward[jnp.newaxis], forward_prob])
        cond_log_likelihoods = jnp.concatenate([initial_log_likelihood[jnp.newaxis], cond_log_likelihoods])

        backward_prob = HMMBase.backward_step(obs_log_probs, trans_log_prob, cond_log_likelihoods)

        return forward_prob * backward_prob

    def e_step(self, X):
        obs_log_probs = self.obs_log_prob(X)
        trans_log_prob = self.trans_log_prob()
        initial_log_prob = self.initial_log_prob()

        return self._e_step(obs_log_probs, trans_log_prob, initial_log_prob)

    @staticmethod
    @jit
    def _viterbi_one_step(a, b):
        """
        :params: a: jnp array, shape (*, batch_size, hidden, hidden)
        :params: b: jnp array, shape (*, batch_size, hidden, hidden)
        """
        if len(a) == 0:
            return a
        return jnp.max(jnp.expand_dims(a, axis=4) + jnp.expand_dims(b, axis=2), axis=3)

    @staticmethod
    @jit
    def _viterbi(initial_log_prob, trans_log_prob, obs_log_prob):
        """
        initial_log_prob: jnp array, shape (hidden)
        trnas_log_prob: jnp array, shape(hidden, hidden)
        obs_log_prob: jnp array, shape(time, batch_size, hidden)
        """
        obs_plus_trans = jnp.expand_dims(obs_log_prob[1:], axis=-1) + jnp.expand_dims(trans_log_prob, axis=(0, 1))   
        obs_plus_initial = jnp.expand_dims(obs_log_prob[0], axis=-1) + jnp.expand_dims(initial_log_prob, axis=(0, 1, 2))
        elem = jnp.concatenate([obs_plus_initial, obs_plus_trans])
        forward = lax.associative_scan(HMMBase._viterbi_one_step, elem)

        obs_plus_trans = jnp.expand_dims(obs_log_prob[1:], axis=2) + jnp.expand_dims(trans_log_prob, axis=(0, 1,))
        elem = jnp.concatenate([obs_plus_trans, jnp.zeros_like(obs_plus_trans[0])[jnp.newaxis]], axis=0)
        backward = lax.associative_scan(HMMBase._viterbi_one_step, elem, reverse=True)

        result = jnp.argmax(forward[:, :, 0] + backward[:, :, :, 0], axis=2)
        return result
        
    def viterbi(self, X):
        initial_log_prob = self.initial_log_prob
        trans_log_prob = self.trans_log_prob
        obs_log_prob = self.obs_log_prob(X)

        return self._viterbi(initial_log_prob, trans_log_prob, obs_log_prob)


        

        


  
