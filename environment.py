import math
import gym
from gym import spaces
import numpy as np
import os


class STAR_RIS(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, num_antennas,
                 num_STAR_elements,
                 num_users,
                 num_elements_T,
                 num_elements_R,
                 num_user_T,
                 num_user_R,
                 channel_est_error=False,
                 AWGN_var=1e-2,
                 channel_noise_var=1e-2,
                 seed=0
                 ):
        super(STAR_RIS, self).__init__()

        self._max_episode_steps = np.inf

        self.M = num_antennas
        self.L = num_STAR_elements
        self.K = num_users
        self.Lt = num_elements_T
        self.Lr = num_elements_R
        self.Kt = num_user_T
        self.Kr = num_user_R
        self.channel_est_error = channel_est_error
        self.AWGN_var = AWGN_var
        self.channel_noise_var = channel_noise_var
        self.seed = seed

        power_size = 2 * self.K

        channel_size = 2 * self.L * self.M + 2 * 2 * self.K * self.L  # H_1 M*L  and h_t, h_2 L * K

        self.action_dim = 2 * self.M * self.K + 2 * self.L  # G and Phi1, Phi2
        self.state_dim = self.action_dim + power_size + channel_size  # action space, channel matrix, T&R matrix, power

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=float)
        self.action_space = spaces.Box(low=-5, high=5, shape=(self.action_dim,), dtype=float)

        self.H1 = None

        # Create G matrix
        self.G = np.random.normal(scale=1, size=(self.M, 2)) + np.random.normal(scale=1, size=(self.M, 2)) * 1j

        # Create channel matrix
        self.h_t = np.random.normal(scale=1, size=(self.L, self.Kr)) + np.random.normal(scale=1,
                                                                                        size=(self.L, self.Kr)) * 1j
        self.h_r = np.random.normal(scale=1, size=(self.L, self.Kt)) + np.random.normal(scale=1,
                                                                                        size=(self.L, self.Kt)) * 1j
        self.H_1 = np.random.normal(scale=1, size=(self.M, self.L)) + np.random.normal(scale=1,
                                                                                       size=(self.M, self.L)) * 1j

        # Create Phi1, Phi2 matrix
        flattenPhi1 = np.zero((self.L, 1))
        flattenPhi1[:self.Lt]
        flattenPhi2 = np.zerp((self.L, 1))
        flattenPhi2[-self.Lr:]
        self.Phi1 = np.diag(flattenPhi1.flatten())
        self.Phi2 = np.diag(flattenPhi2.flatten())

        self.X = np.random.normal(scale=1, size=(self.K, 1)) + np.random.normal(scale=1, size=(self.K, 1)) * 1j

        self.state = None

        self.episode_t = None

        self.reward = 0

        self.seed(seed)

        self.data_rate_list_R = np.zeros(self.KR)
        self.data_rate_list_T = np.zeros(self.KT)

        self.t = 0

        # self.reset()

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        self.episode_t = 0

        self.H_1 = np.random.normal(0, np.sqrt(0.5), (self.L, self.M)) + 1j * np.random.normal(0, np.sqrt(0.5),
                                                                                               (self.L, self.M))

        self.h_t = np.random.normal(0, np.sqrt(0.5), (self.L, 1)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.L, 1))
        self.h_r = np.random.normal(0, np.sqrt(0.5), (self.L, 1)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.L, 1))

        init_action_G = np.hstack((np.real(self.G.reshape(1, -1)), np.imag(self.G.reshape(1, -1))))
        init_action_Phi1 = np.hstack(
            (np.real(np.diag(self.Phi1)).reshape(1, -1), np.imag(np.diag(self.Phi1)).reshape(1, -1)))
        init_action_Phi2 = np.hstack(
            (np.real(np.diag(self.Phi2)).reshape(1, -1), np.imag(np.diag(self.Phi2)).reshape(1, -1)))

        init_action = np.hstack((init_action_G, init_action_Phi1, init_action_Phi2))

        power_t = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2

        receive = self.h_r.T @ self.Phi1 @ self.H_1 @ self.G + self.h_t.T @ self.Phi2 @ self.H_2 @ self.G

        power_r = np.linalg.norm(receive, axis=0).reshape(1, -1) ** 2

        H_1_real, H_1_imag = np.real(self.H_1).reshape(1, -1), np.imag(self.H_1).reshape(1, -1)
        h_t_real, h_t_imag = np.real(self.h_t).reshape(1, -1), np.imag(self.h_t).reshape(1, -1)
        h_r_real, h_r_imag = np.real(self.h_r).reshape(1, -1), np.imag(self.h_r).reshape(1, -1)

        self.state = np.hstack(
            (init_action, power_t, power_r, H_1_real, H_1_imag, h_t_real, h_t_imag, h_r_real, h_r_imag))

        return self.state

    def step(self, action):
        self.episode_t += 1

        action = action.reshape(1, -1)

        G_real = action[:, :self.M * self.K]
        G_imag = action[:, self.M * self.K:2 * self.M * self.K]

        Phi1_real = action[:, -4 * self.L:-3 * self.L]
        Phi1_imag = action[:, -3 * self.L:-2 * self.L]

        Phi2_real = action[:, -2 * self.L:-self.L]
        Phi2_imag = action[:, -self.L:]

        self.G = G_real.reshape(self.M, self.K) + 1j * G_imag.reshape(self.M, self.K)
        self.Phi1 = np.diag((Phi1_real + 1j * Phi1_imag))
        self.Phi2 = np.diag((Phi2_real + 1j * Phi2_imag))

        power_t = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2

        receive = self.h_r.T @ self.Phi1 @ self.H_1 @ self.G + self.h_t.T @ self.Phi2 @ self.H_2 @ self.G

        power_r = np.linalg.norm(receive, axis=0).reshape(1, -1) ** 2

        H_1_real, H_1_imag = np.real(self.H_1).reshape(1, -1), np.imag(self.H_1).reshape(1, -1)
        h_t_real, h_t_imag = np.real(self.h_t).reshape(1, -1), np.imag(self.h_t).reshape(1, -1)
        h_r_real, h_r_imag = np.real(self.h_r).reshape(1, -1), np.imag(self.h_r).reshape(1, -1)

        self.state = np.hstack((action, power_t, power_r, H_1_real, H_1_imag, h_t_real, h_t_imag, h_r_real, h_r_imag))

        self.calculate_data_rate()
        self.sum_rate = sum(self.data_rate_list_R) + sum(self.data_rate_list_T)
        self.reward = self.sum_rate

        if (self.episode_t == self._max_episode_steps):
            done = True
        else:
            done = False

        info = {}

        return self.state, self.reward, done, info

    def calculate_data_rate(self):

        for k_T in range(self.Kt):
            # Computing cascaded channels
            h_k_t = self.h_r[:, k_T].conj().T

            Signal_power_k_T = abs(h_k_t @ self.Phi1 @ self.H1 @ self.G @ self.X[:, k_T]) ** 2  # calculate signal power
            Interference_power_k_T = 0 - abs(
                h_k_t @ self.Phi1 @ self.H1 @ self.G @ self.X[:, k_T]) ** 2  # calculate interference power
            for j_T in range(self.K):
                Interference_power_k_T += abs(h_k_t @ self.Phi1 @ self.H1 @ self.G @ self.X[:, j_T]) ** 2

            SINR_k_T = Signal_power_k_T / (Interference_power_k_T + self.noise_power)
            data_rate_k_T = self.B * math.log((1 + SINR_k_T), 2)  # calculate data rate
            self.data_rate_list_R[k_T] = data_rate_k_T  # saving data rate

        for k_R in range(self.Kr):
            # Computing cascaded channels
            h_k_r = self.h_r[:, k_R].conj().T

            Signal_power_k_R = abs(h_k_r @ self.Phi2 @ self.H1 @ self.G @ self.X[:, k_R]) ** 2  # calculate signal power
            Interference_power_k_R = 0 - abs(
                h_k_r @ self.Phi2 @ self.H1 @ self.G @ self.X[:, k_R]) ** 2  # calculate interference power
            for j_R in range(self.K):
                Interference_power_k_R += abs(h_k_r @ self.Phi2 @ self.H1 @ self.G @ self.X[:, j_R]) ** 2

            SINR_k_R = Signal_power_k_R / (Interference_power_k_R + self.noise_power)
            data_rate_k_R = self.B * math.log((1 + SINR_k_R), 2)  # calculate data rate
            self.data_rate_list_R[k_R] = data_rate_k_R  # saving data rate

    def close(self):
        pass