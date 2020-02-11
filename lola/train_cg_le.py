"""
Training funcion for the Coin Game.
"""
import os
import numpy as np
import tensorflow as tf
import pdb

from . import logger

from .corrections import *
from .networks import *
from .utils import *


def update(mainPN, lr, final_delta_1_v, final_delta_2_v):
    update_theta_1 = mainPN[0].setparams(
        mainPN[0].getparams() + lr * np.squeeze(final_delta_1_v))
    update_theta_2 = mainPN[1].setparams(
        mainPN[1].getparams() + lr * np.squeeze(final_delta_2_v))


def update_single(PN, lr, final_delta_v):
    update_theta_1 = PN.setparams(
        PN.getparams() + lr * np.squeeze(final_delta_v))


def clone_update(mainPN_clone):
    for i in range(2):
        mainPN_clone[i].log_pi_clone = tf.reduce_mean(
            mainPN_clone[i].log_pi_action_bs)
        mainPN_clone[i].clone_trainer = \
            tf.train.GradientDescentOptimizer(learning_rate=0.1)
        mainPN_clone[i].update = mainPN_clone[i].clone_trainer.minimize(
            -mainPN_clone[i].log_pi_clone, var_list=mainPN_clone[i].parameters)


def train(env, *, num_episodes, trace_length, batch_size,
          corrections, opp_model, grid_size, gamma, hidden, bs_mul, lr,
          welfare0, welfare1, punish=False,
          mem_efficient=True, num_punish_episodes=1000):
    #Setting the training parameters
    batch_size = batch_size #How many experience traces to use for each training step.
    trace_length = trace_length #How long each experience trace will be when training

    y = gamma
    num_episodes = num_episodes #How many episodes of game environment to train network with.
    load_model = False #Whether to load a saved model.
    path = "./drqn" #The path to save our model to.
    n_agents = env.NUM_AGENTS
    total_n_agents = n_agents
    h_size = [hidden] * total_n_agents
    max_epLength = trace_length+1 #The max allowed length of our episode.
    summary_len = 20 #Number of episodes to periodically save for analysis

    tf.reset_default_graph()
    mainPN = []
    mainPN_step = []
    coopPN = []
    coopPN_step = []
    punishPN = []
    punishPN_step = []
    agent_list = np.arange(total_n_agents)
    for agent in range(total_n_agents):
        mainPN.append(
            Pnetwork('main' + str(agent), h_size[agent], agent, env,
                trace_length=trace_length, batch_size=batch_size,))
        mainPN_step.append(
            Pnetwork('main' + str(agent), h_size[agent], agent, env,
                trace_length=trace_length, batch_size=batch_size,
                reuse=True, step=True))

    # Clones of the opponents
    if opp_model:
        mainPN_clone = []
        for agent in range(total_n_agents):
            mainPN_clone.append(
                Pnetwork('clone' + str(agent), h_size[agent], agent, env,
                         trace_length=trace_length, batch_size=batch_size))

    if punish:  # Initialize punishment networks and networks for tracking cooperative updates
        punishPN = Pnetwork('punish' + str(0), h_size[0], 0, env,
                     trace_length=trace_length, batch_size=batch_size, )
        punishPN_step = Pnetwork('punish' + str(0), h_size[0], 0, env,
                     trace_length=trace_length, batch_size=batch_size,
                     reuse=True, step=True)
        coopPN = Pnetwork('coop' + str(1), h_size[1], 1, env,
                     trace_length=trace_length, batch_size=batch_size, )
        coopPN_step = Pnetwork('coop' + str(1), h_size[1], 1, env,
                     trace_length=trace_length, batch_size=batch_size,
                     reuse=True, step=True)

    if not mem_efficient:
        cube, cube_ops = make_cube(trace_length)
    else:
        cube, cube_ops = None, None

    if not opp_model:
        corrections_func(mainPN, batch_size, trace_length, corrections, cube)
        if punish:
            corrections_func_single(coopPN, batch_size, trace_length)
            corrections_func_single(punishPN, batch_size, trace_length)
    else:
        corrections_func([mainPN[0], mainPN_clone[1]],
                         batch_size, trace_length, corrections, cube)
        corrections_func([mainPN[1], mainPN_clone[0]],
                         batch_size, trace_length, corrections, cube)
        corrections_func([mainPN[1], mainPN_clone[0]],
                         batch_size, trace_length, corrections, cube)
        
        clone_update(mainPN_clone)

    init = tf.global_variables_initializer()
    # saver = tf.train.Saver(max_to_keep=5)

    trainables = tf.trainable_variables()

    #create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    aList = []

    total_steps = 0
    has_defected = False
    time_to_punish = False

    # Make a path for our model to be saved in.
    if not os.path.exists(path):
        os.makedirs(path)

    episodes_run = np.zeros(total_n_agents)
    episodes_run_counter =  np.zeros(total_n_agents)
    episodes_reward = np.zeros((total_n_agents, batch_size))
    episodes_actions = np.zeros((total_n_agents, env.NUM_ACTIONS))

    pow_series = np.arange(trace_length)
    discount = np.array([pow(gamma, item) for item in pow_series])
    discount_array = gamma**trace_length / discount
    discount = np.expand_dims(discount, 0)
    discount_array = np.reshape(discount_array,[1,-1])

    with tf.Session() as sess:
        # if load_model == True:
        #     print( 'Loading Model...')
        #     ckpt = tf.train.get_checkpoint_state(path)
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        sess.run(init)
        if not mem_efficient:
            sess.run(cube_ops)

        sP = env.reset()
        updated =True
        for i in range(num_episodes):
            a0_defected = False
            a1_defected = False
            episodeBuffer = []
            for ii in range(n_agents):
                episodeBuffer.append([])
            np.random.shuffle(agent_list)
            if n_agents  == total_n_agents:
                these_agents = range(n_agents)
            else:
                these_agents = sorted(agent_list[0:n_agents])

            #Reset environment and get first new observation
            sP = env.reset()
            s = sP

            trainBatch0 = [[], [], [], [], [], []]
            trainBatch1 = [[], [], [], [], [], []]
            coopTrainBatch1 = [[], [], [], [], [], []]

            d = False
            rAll = np.zeros((4))
            aAll = np.zeros((env.NUM_ACTIONS * 2))
            j = 0

            # ToDo: need to track lstm states for main and punish nets
            lstm_state = []
            for agent in these_agents:
                episodes_run[agent] += 1
                episodes_run_counter[agent] += 1
                lstm_state.append(np.zeros((batch_size, h_size[agent]*2)))
            if punish: 
                lstm_coop_state = np.zeros((batch_size, h_size[1]*2))

            while j < max_epLength:
                lstm_state_old = lstm_state
                if punish: lstm_coop_state_old = lstm_coop_state
                j += 1
                a_all = []
                lstm_state = []
                lstm_punish_state = []
                for agent_role, agent in enumerate(these_agents):
                    # Actual actions and lstm states
                    if punish and time_to_punish and agent == 0: # Assuming only agent 0 punishes
                        a, lstm_punish_s = sess.run(
                            [
                                punishPN_step.predict,
                                punishPN_step.lstm_state_output
                            ],
                            feed_dict={
                                punishPN_step.state_input: s,
                                punishPN_step.lstm_state: lstm_state_old[agent]
                            }
                        )
                        lstm_punish_state.append(lstm_s)
                        a_all.append(a)
                    else:
                        a, lstm_s = sess.run(
                            [
                                mainPN_step[agent].predict,
                                mainPN_step[agent].lstm_state_output
                            ],
                            feed_dict={
                                mainPN_step[agent].state_input: s,
                                mainPN_step[agent].lstm_state: lstm_state_old[agent]
                            }
                        )
                        lstm_state.append(lstm_s)
                        a_all.append(a)

                    # Cooperative actions and lstm states
                    if punish and agent == 1: # Assuming only agent 1 can be non-cooperative
                        a_coop, lstm_s_coop = sess.run(
                            [
                                coopPN_step.predict,
                                coopPN_step.lstm_state_output
                            ],
                            feed_dict={
                                coopPN_step.state_input: s,
                                coopPN_step.lstm_state: lstm_coop_state_old
                            }
                        )
                        lstm_coop_state = lstm_s_coop
                        a_coop_all = a_coop

                # ToDo: make sure the policies which are being compared are deterministic (i.e. account for
                # ToDo: random seed where necessary
                if punish and not time_to_punish and not has_defected:
                    if np.array_equal(a_coop, a_all[1]):
                        has_defected = False
                    else:
                        has_defected = True

                # ToDo: need separate trainBatch for punishment?
                # Add obs for policy network
                trainBatch0[0].append(s)
                trainBatch1[0].append(s)
                trainBatch0[1].append(a_all[0])
                trainBatch1[1].append(a_all[1])

                # Add obs for coop network
                # ToDo: update coop only if has_defected == False?
                if punish:
                    coopTrainBatch1[0].append(s)
                    coopTrainBatch1[1].append(a_all[1])

                a_all = np.transpose(np.vstack(a_all))

                s1P,r,d = env.step(actions=a_all)
                s1 = s1P

                trainBatch0[2].append(r[0])
                trainBatch1[2].append(r[1])
                trainBatch0[3].append(s1)
                trainBatch1[3].append(s1)
                trainBatch0[4].append(d)
                trainBatch1[4].append(d)
                trainBatch0[5].append(lstm_state[0])
                trainBatch1[5].append(lstm_state[1])

                if punish: 
                    coopTrainBatch1[2].append(s1)  # Coop train batch doesn't have entry for rewards b/c welfare function
                    coopTrainBatch1[3].append(d)
                    coopTrainBatch1[4].append(lstm_coop_state)

                total_steps += 1
                for agent_role, agent in enumerate(these_agents):
                    episodes_reward[agent] += r[agent_role]

                for index in range(batch_size):
                    r_pb = [r[0][index], r[1][index]]
                    if np.array(r_pb).any():
                        if r_pb[0] == 1 and r_pb[1] == 0:
                            rAll[0] += 1
                        elif r_pb[0] == 0 and r_pb[1] == 1:
                            rAll[1] += 1
                        elif r_pb[0] == 1 and r_pb[1] == -2:
                            rAll[2] += 1
                        elif r_pb[0] == -2 and r_pb[1] == 1:
                            rAll[3] += 1

                aAll[a_all[0]] += 1
                aAll[a_all[1] + 4] += 1
                s_old = s
                s = s1
                sP = s1P
                if d.any():
                    break

            jList.append(j)
            rList.append(rAll)
            aList.append(aAll)

            # training after one batch is obtained
            sample_return0 = np.reshape(
                get_monte_carlo(trainBatch0[2], y, trace_length, batch_size),
                [batch_size, -1])
            sample_return1 = np.reshape(
                get_monte_carlo(trainBatch1[2], y, trace_length, batch_size),
                [batch_size, -1])
            if punish and time_to_punish:
                sample_return0 = -sample_return1
            else:
                sample_return0 = welfare0(sample_return0, sample_return1)
                sample_return1 = welfare1(sample_return1, sample_return0)

            # need to multiple with
            pow_series = np.arange(trace_length)
            discount = np.array([pow(gamma, item) for item in pow_series])

            sample_reward0 = discount * np.reshape(
                trainBatch0[2] - np.mean(trainBatch0[2]), [-1, trace_length])
            sample_reward1 = discount * np.reshape(
                trainBatch1[2]- np.mean(trainBatch1[2]), [-1, trace_length])
            # ToDo: Check that calculation of rewards and returns are correct given how they're used
            if punish and time_to_punish:
                sample_reward0 = -sample_reward1
            else:
                sample_reward0 = welfare0(sample_reward0, sample_reward1)
                sample_reward1 = welfare1(sample_reward1, sample_reward0)

            state_input0 = np.concatenate(trainBatch0[0], axis=0)
            state_input1 = np.concatenate(trainBatch1[0], axis=0)
            actions0 = np.concatenate(trainBatch0[1], axis=0)
            actions1 = np.concatenate(trainBatch1[1], axis=0)

            last_state = np.reshape(
                np.concatenate(trainBatch1[3], axis=0),
                [batch_size, trace_length, env.ob_space_shape[0],
                 env.ob_space_shape[1], env.ob_space_shape[2]])[:,-1,:,:,:]

            # ToDo: should be option for updating punishPN if punish==True
            value_0_next, value_1_next = sess.run(
                [mainPN_step[0].value, mainPN_step[1].value],
                feed_dict={
                    mainPN_step[0].state_input: last_state,
                    mainPN_step[1].state_input: last_state,
                    mainPN_step[0].lstm_state: lstm_state[0],
                    mainPN_step[1].lstm_state: lstm_state[1],
                })

            if punish:
                value_coop_next = sess.run(
                    coopPN_step.value,
                    feed_dict={coopPN_step.state_input: last_state,
                               coopPN_step.lstm_state: lstm_coop_state}
                )
            # if opp_model:
            #     ## update local clones
            #     update_clone = [mainPN_clone[0].update, mainPN_clone[1].update]
            #     feed_dict = {
            #         mainPN_clone[0].state_input: state_input1,
            #         mainPN_clone[0].actions: actions1,
            #         mainPN_clone[0].sample_return: sample_return1,
            #         mainPN_clone[0].sample_reward: sample_reward1,
            #         mainPN_clone[1].state_input: state_input0,
            #         mainPN_clone[1].actions: actions0,
            #         mainPN_clone[1].sample_return: sample_return0,
            #         mainPN_clone[1].sample_reward: sample_reward0,
            #         mainPN_clone[0].gamma_array: np.reshape(discount,[1,-1]),
            #         mainPN_clone[1].gamma_array: np.reshape(discount,[1,-1]),
            #     }
            #     num_loops = 50 if i == 0 else 1
            #     for i in range(num_loops):
            #         sess.run(update_clone, feed_dict=feed_dict)

            #     theta_1_vals = mainPN[0].getparams()
            #     theta_2_vals = mainPN[1].getparams()
            #     theta_1_vals_clone = mainPN_clone[0].getparams()
            #     theta_2_vals_clone = mainPN_clone[1].getparams()

            #     if len(rList) % summary_len == 0:
            #         print('params check before optimization')
            #         print('theta_1_vals', theta_1_vals)
            #         print('theta_2_vals_clone', theta_2_vals_clone)
            #         print('theta_2_vals', theta_2_vals)
            #         print('theta_1_vals_clone', theta_1_vals_clone)
            #         print('diff between theta_1 and theta_2_vals_clone',
            #             np.linalg.norm(theta_1_vals - theta_2_vals_clone))
            #         print('diff between theta_2 and theta_1_vals_clone',
            #             np.linalg.norm(theta_2_vals - theta_1_vals_clone))

            # Update policy networks
            if punish and time_to_punish:
                network_to_update = punishPN
            else:
                network_to_update = mainPN

            feed_dict={
                network_to_update[0].state_input: state_input0,
                network_to_update[0].sample_return: sample_return0,
                network_to_update[0].actions: actions0,
                network_to_update[1].state_input: state_input1,
                network_to_update[1].sample_return: sample_return1,
                network_to_update[1].actions: actions1,
                network_to_update[0].sample_reward: sample_reward0,
                network_to_update[1].sample_reward: sample_reward1,
                network_to_update[0].gamma_array: np.reshape(discount, [1, -1]),
                network_to_update[1].gamma_array: np.reshape(discount, [1, -1]),
                network_to_update[0].next_value: value_0_next,
                network_to_update[1].next_value: value_1_next,
                network_to_update[0].gamma_array_inverse:
                    np.reshape(discount_array, [1, -1]),
                network_to_update[1].gamma_array_inverse:
                    np.reshape(discount_array, [1, -1]),
            }
            # if opp_model:
            #     feed_dict.update({
            #         mainPN_clone[0].state_input:state_input1,
            #         mainPN_clone[0].actions: actions1,
            #         mainPN_clone[0].sample_return: sample_return1,
            #         mainPN_clone[0].sample_reward: sample_reward1,
            #         mainPN_clone[1].state_input:state_input0,
            #         mainPN_clone[1].actions: actions0,
            #         mainPN_clone[1].sample_return: sample_return0,  # This is what forms the target of the PNetwork
            #         mainPN_clone[1].sample_reward: sample_reward0,
            #         mainPN_clone[0].gamma_array: np.reshape(discount,[1,-1]),
            #         mainPN_clone[1].gamma_array:  np.reshape(discount,[1,-1]),
            #     })
            values, _, _, update1, update2 = sess.run(
              [
                  network_to_update[0].value,
                  network_to_update[0].updateModel,
                  network_to_update[1].updateModel,
                  network_to_update[0].delta,
                  network_to_update[1].delta,
              ],
              feed_dict=feed_dict)

            update(network_to_update, lr, update1 / bs_mul, update2 / bs_mul)
            updated = True
            print('update params')
            
            # Update cooperative policy network
            if punish and not time_to_punish:
                feed_dict = {
                    coopPN.state_input: state_input1,
                    coopPN.sample_return: sample_return0,  # Assuming returns for agent 0 are given by welfare fn
                    coopPN.actions: actions1,
                    coopPN.sample_reward: sample_reward0,  # Assuming returns for agent 0 are given by welfare fn
                    coopPN.gamma_array: np.reshape(discount, [1, -1]),
                    coopPN.next_value: value_coop_next,
                    coopPN.gamma_array_inverse:
                        np.reshape(discount_array, [1, -1]),
                }
                values, _, update_coop = sess.run(
                    [
                        coopPN.value,
                        coopPN.updateModel,
                        coopPN.delta
                    ],
                    feed_dict=feed_dict)
                update_single(coopPN, lr, update_coop) # ToDo: change update to accomodate None

            episodes_run_counter[agent] = episodes_run_counter[agent] * 0
            episodes_actions[agent] = episodes_actions[agent] * 0
            episodes_reward[agent] = episodes_reward[agent] * 0

            # Update punishment tracking
            if punish and time_to_punish:
                punish_episode_counter -= 1
                if punish_counter == 0:
                  time_to_punish = False
            else:
                if has_defected:
                    time_to_punish = True
                    punish_episode_counter = num_punish_episodes

            if len(rList) % summary_len == 0 and len(rList) != 0 and updated:
                updated = False
                print(total_steps, 'reward', np.sum(rList[-summary_len:], 0))
                rlog = np.sum(rList[-summary_len:], 0)
                for ii in range(len(rlog)):
                    logger.record_tabular('rList['+str(ii)+']', rlog[ii])
                logger.dump_tabular()
                logger.info('')
