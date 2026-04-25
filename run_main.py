from maze_env import Maze
from RL_brainsample_qlearning import rlalgorithm as rlalg1
from RL_brainsample_sarsa import rlalgorithm as rlalg2
# from RL_brainsample_PI import rlalgorithm as rlalg3
from RL_brainsample_expsarsa import rlalgorithm as rlalg4
from RL_brainsample_EligTrace import rlalgorithm as rlalg5
from RL_brainsample_wrong import rlalgorithm as rlalg7
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import time
import warnings
from IPython.display import display

DEBUG=1
def debug(debuglevel, msg, **kwargs):
    if debuglevel <= DEBUG:
        if 'printNow' in kwargs:
            if kwargs['printNow']:
                print(msg) 
        else:
            print(msg) 


def plot_rewards(experiments, window=100,save_fig=False, savefileprefix='', logPlot=False):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    color_list=['blue','red','green','black','purple', 'orange', 'cyan', 'olive', 'pink']
    light_color_list=['lightblue','lightcoral','lightgreen', 'darkgrey', 'magenta', 'moccasin', 'lightcyan', 'khaki', 'plum']
    label_list=[]
    for i, (name, env, RL, data) in enumerate(experiments):
        x_values=range(len(data['global_reward']))
        label_list.append(RL.display_name)
        y_values=data['global_reward']
        plt.plot(x_values, y_values, c=light_color_list[i],label=label_list[-1])
        plt.legend(label_list)
    if len(x_values) >= window : 
        for i, (name, env, RL, data) in enumerate(experiments):
            x_values=range(window, 
                    len(data['med_rew_window'])+window)
            y_values=data['med_rew_window']
            plt.plot(x_values, y_values,
                    c=color_list[i])
    plt.title("Summed Reward", fontsize=16)
    plt.xlabel("Episode", fontsize=16)
    plt.ylabel("Reward", fontsize=16)
    plt.tick_params(axis='both', which='major',
                    labelsize=14)

    plt.suptitle("Task " + str(usetask) + " : " + RLargsStr, fontsize=20)

    # plt.figure()
    plt.subplot(122)
    label_list=[]
    for i, (name, env, RL, data) in enumerate(experiments):
        x_values=range(len(data['ep_length']))
        label_list.append(RL.display_name)
        y_values=np.log(data['ep_length']) if logPlot else data['ep_length']
        plt.plot(x_values, y_values, c=light_color_list[i],label=label_list[-1])
        plt.legend(label_list)
    plt.title("Path Length", fontsize=16)
    plt.xlabel("Episode", fontsize=16)
    plt.ylabel(f"{'log(Length)' if logPlot else 'Length'}", fontsize=16)
    plt.tick_params(axis='both', which='major',
                    labelsize=14)
    if save_fig:
        plt.savefig(savefileprefix + '.png')
    # plt.show()

# FUTURE TODO: over top of the maze image, 

def plot_Trajectories(experiment,save_fig=False, savefileprefix=''):
    '''For one Experiment, create a plot of the trajectories taken in each episode for the experiment.
    The line colors fade from red early on to green by the end and are partially transparent so that overlapping
    can be seen over the entire training run.
    We use the data['trajectories_per_episode'] list of tuples in the form ((state_column, state_row),action) 
    Create a plot title using the experiment name.
    '''
    name, env, RL, data = experiment
    plt.figure(figsize=(8, 8))
    plt.title(f'Trajectories taken by {RL.display_name} on Task {usetask}', fontsize=16)
    plt.xlim(0, env.MAZE_W)
    # plt.ylim(env.MAZE_H,0)
    plt.ylim(0,env.MAZE_H)
    plt.xlabel('Column', fontsize=16)
    plt.ylabel('Row', fontsize=16)
    plt.gca().set_aspect('equal', adjustable='box')
    # show gride lines for every integer unit
    plt.xticks(np.arange(0, env.MAZE_W+1, 1.0))
    plt.yticks(np.arange(0, env.MAZE_H+1, 1.0))
    # plt.tick_params(axis='both', which='both', labelsize=14)
    plt.grid(True)
    plt.gca().invert_yaxis() # because row 0 is at the top in the maze env
    # loaded_image = plt.imread("maze_background.png")
    # loaded_image = Image.open("maze_background.png")
    # plt.imshow(loaded_image, extent=[0, env.MAZE_W, 0, env.MAZE_H])
    num_episodes = len(data['trajectories_per_episode'])
    for ep in range(num_episodes):
        portion = ep/num_episodes
        jitter = 0.5* (portion - .5)
        traj = data['trajectories_per_episode'][ep]
        num_steps = len(traj)
        color = (1-portion, portion, 0.3)
        for step in range(num_steps-1):
            (col1, row1), action = traj[step]
            (col2, row2), _ = traj[step+1]
            plt.arrow(col1 + 0.5+jitter, row1 + 0.5 + jitter, col2 - col1+jitter, row2 - row1,
                      head_width=0.2, head_length=0.2, fc=color, ec=color, alpha=0.3, width=.1)

    if save_fig:
        plt.savefig(savefileprefix + '.png')
    # plt.show()  





def saveData(name, env, RL, data, savefileprefix):
    filename=savefileprefix + '.pkl'
    print(f'Saving experiment {name} to {filename}')
    # Save experiment data to new file
    with open(filename, 'wb') as f:
        pickle.dump((RL,data), f)
        f.close()

def loadData(filename):
    pickle.load(open('data/'+filename,'r'))


def value_iteration(env, RL, data, sweeps):
    '''Run first pass through all states with a deterministic algorithm to fill the Q(s,a) table
    '''    

    global_reward = np.zeros(sweeps)
    data['global_reward']=global_reward

    for swp in range(sweeps):
        t=0
        if swp == 0:
            state = env.reset(value = 0)
        else:
            state = env.reset()

        allS_count, s, s_count = RL.count_state(str(state))

        if swp % printEveryNth == 0:
            debug(1, f'\n---------\nQTable after {swp-1} sweeps:\n . |Q.S|={allS_count}/{max_states} maxq={RL.maxq} delta={RL.delta}')
            # debug(2, f'{display(RL.q_table)}')
            debug(2, f'{RL.q_table.describe()}')
        debug(2, f'|Q.S|={allS_count} #s({s})={s_count}') 
        debug(2,'state(swp:{},t:{})={}'.format(swp, t, state))

        # The main loop of the training on an episode
        # RL choose action based on state
        action = RL.choose_action(str(state),policy=4)

        if allS_count < max_states:
            while True:

                # RL take action and get next state and reward
                state_, reward, done = env.step(action)
                allS_count, s, s_count =RL.count_state(str(state_))
                debug(2, f'|Q.S|={allS_count} #s({s})={s_count}') 

                global_reward[swp] += reward

                debug(2,'state(swp:{},t:{})={}'.format(swp, t, state))
                debug(2,'reward={:.3f} return_t={:.3f} Mean50={:.3f}'.format(reward, global_reward[swp],np.mean(global_reward[-50:])))

                state_, action_ = RL.learn(str(state), action, reward, str(state_), policy=4)

                # action = RL.choose_action(str(state),policy=4)
                action = action_
                state = state_

                # break while loop when end of this episode
                if done:
                    # RL learn from this trajectory
                    break
                else:
                    t=t+1

            debug(2,"({}) Swp {} Length={} Summed Reward={:.3} ".format(RL.display_name,swp, t,  global_reward[swp],global_reward[swp]),printNow=(swp%printEveryNth==0))
            
        else:
            for i in range(int(max_states/2)):
                state = RL.q_table.keys()[np.random.randint(0, len(RL.q_table.keys()))]
                action = RL.choose_action(str(state),policy=4)

                state_, reward, done = env.step(action)
                allS_count, s, s_count =RL.count_state(str(state_))
                debug(2, f'|Q.S|={allS_count} #s({s})={s_count}') 

                global_reward[swp] += reward

                debug(2,'state(swp:{},t:{})={}'.format(swp, t, state))
                debug(2,'reward={:.3f} return_t={:.3f} Mean50={:.3f}'.format(reward, global_reward[swp],np.mean(global_reward[-50:])))
                RL.learn(str(state), action, reward, str(state_), policy=4)


    print('Algorithm {} completed'.format(RL.display_name))
    env.destroy()


def update(env, RL, data, episodes=100, window=10, showRender=True, renderEveryNth=100, learn_from_transitions=True, sim_speed=0.1, *args, **kwargs):
    '''Main loop for RL '''
    global_reward = np.zeros(episodes)
    data['global_reward']=global_reward
    ep_length = np.zeros(episodes)
    data['ep_length']=ep_length
    if episodes >= window:
        med_rew_window = np.zeros(episodes-window)
        var_rew_window = np.zeros(episodes)
    else:
        med_rew_window = []
        var_rew_window = []

    if save_trajectories:
        data['trajectories_per_episode'] = []
        print('Saving trajectories per episode')
    data['med_rew_window'] = med_rew_window
    data['var_rew_window'] = var_rew_window

    for episode in range(episodes):  
        t=0
        renderNow = showRender and episode>=1 and (episode % renderEveryNth)==0

        ''' initial state
            Note: the state is represented as two pairs of 
            coordinates, for the bottom left corner and the 
            top right corner of the agent square.
        '''
        if episode == 0:
            state = env.reset(value = 0, renderNow=renderNow)
        else:
            state = env.reset(renderNow=renderNow)

        debug(2,'state(ep:{},t:{})={}'.format(episode, t, state))

        if(renderNow):
            print('Rendering Now Alg:{} Ep:{}/{} at speed:{}'.format(RL.display_name, episode, episodes, sim_speed))

        # The main loop of the training on an episode
        # RL choose action based on state
        action = RL.choose_action(str(state))

        while True:
            # fresh env
            if(renderNow):
                env.render(sim_speed)

            # RL take action and get next state and reward
            state_, reward, done = env.step(action, renderNow)
            global_reward[episode] += reward
            debug(2,'state(ep:{},t:{})={}'.format(episode, t, state))
            debug(2,'   reward={:.3f} return_t={:.3f} Mean50={:.3f}'.format(reward, global_reward[episode],np.mean(global_reward[-50:])))

            # if learn_from_transitions:
            # RL learn from this transition
            # and determine next state and action
            state, action =  RL.learn(str(state), action, reward, str(state_), **kwargs)
            state = str(state_)
            if action is None:
                # If action is None, it means the algorithm does not return an action
                # This is the case for Monte Carlo and Value Iteration algorithms
                action = RL.choose_action(str(state))
            # else:
            #     # Learn from the entire trajectory, just send the information to learn for now to build up the data
            #     RL.learn(str(state), 
            #              action, 
            #              reward, 
            #              str(state_), 
            #              learn_from_transitions=learn_from_transitions, 
            #              **kwargs)
            #     action = RL.choose_action(str(state))
            #     state = state_

            # break while loop when end of this episode
            if done:
                break
            else:
                t=t+1

        debug(1,"({}) Ep {} Length={} Summed Reward={:.3} ".format(RL.display_name,episode, t,  global_reward[episode]),printNow=(episode%printEveryNth==0))

        #save data about length of the episode
        ep_length[episode]=t
        if save_trajectories:
            data['trajectories_per_episode'].append(env.get_trajectory_data())

        if(episode>=window):
            med_rew_window[episode-window] = np.median(global_reward[episode-window:episode])
            var_rew_window[episode-window] = np.var(global_reward[episode-window:episode])
            debug(1,"    Med-{}={:.3f} Var-{}={:.3f}".format(
                    window,
                    med_rew_window[episode-window],
                    window,
                    var_rew_window[episode-window]),
                printNow=(episode%printEveryNth==0))
    print('Algorithm {} completed'.format(RL.display_name))
    env.destroy()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="The frame\.append method is deprecated.*")
    sim_speed = 0.001 # .1 .05 .001 change this to adjust speed up rendered run
    
    import os
    os.chdir(os.path.dirname(__file__))

    #Which algorithms to run, just as many as needed
    runalg1=1; # QLearning - comparable to SARSA
    runalg2=1; # SARSA - comparable to QLearning
    runalg3=0; # (very optional) Exact PolicyIteration or Value Iteration?
    runalg4=1; #Expected SARSA 
    runalg5=1; # TD(Lambda) 
    runalg7=0 # Wrong - given to students as a demo
    runalg8=0; # Monte Carlo algorithm

    runalg1opt=False; # Dev - not working - QLearning with optimistic initiliaziation (nah, 0 is already optimistic

    #Which task to run, select just one
    usetask = int(os.environ.get('TASK_NUM', '1')) # 1,2,3

    #Example Short Fast start parameters for Debugging
    showRender=False # True means renderEveryNth episode only, False means don't render at all
    save_trajectories = False
    episodes=1500 #was 101 and 501 and 1001
    renderEveryNth=5000 # was 10 50 100 and 150 
    printEveryNth=500 # was 5 and 10 and 25 and 50
    window = 10 #size of window for running median, variance calculations # was 10 50
    do_plot_rewards=True
    do_save_figure=True
    VI_sweeps = 1000
    do_save_data=False

    #Optional, override the default RL init arguments
    RLargs = {
            'learning_rate':0.01, 
            'reward_decay':0.9,
            'e_greedy':0.1
            }
    RLargsStr = ''
    # for key in RLargs:
        # RLargsStr += f" {key}={str(RLargs[key])}"

    RLargsStr = 'LR{learning_rate}_gamma{reward_decay}_eps{e_greedy}'.format(**RLargs)

    # Some arguments use for running the experiments
    EXPargs = {'sim_speed':sim_speed,
                'window':window, 
                'episodes':episodes, 
                'showRender':showRender, 
                'renderEveryNth':renderEveryNth, 
                'printEveryNth':printEveryNth, 
                'do_plot_rewards':do_plot_rewards, 
                'do_save_figure':do_save_figure, 
                'VI_sweeps':VI_sweeps, 
                'do_save_data':do_save_data}

    #Example Full Run, you may need to run longer
    #showRender=False
    #episodes=1000
    #renderEveryNth=100
    #printEveryNth=20
    #window=100
    #do_plot_rewards=True
    #do_plot_length=True

    if(len(sys.argv)>1):
        episodes = int(sys.argv[1])
    if(len(sys.argv)>2):
        showRender = sys.argv[2] in ['true','True','T','t']
    if(len(sys.argv)>3):
        datafile = sys.argv[3]


    # Task Specifications
    # point [0,0] is the top left corner
    # point [x,y] is x columns over and y rows down
    # range of x and y is [0,9]
    # agentXY=[0,0] # Agent start position [column, row]
    # goalXY=[4,4] # Target position, terminal state

    #Task 1
    if usetask == 1:
        # S25 Task 1
        agentXY=[2,7] # Agent start position
        goalXY=[7, 8] # Target position, terminal state
        wall_shape=np.array([[4,9], [4,8], [4,7], [4,6], [5,1],[4,1], [1,1],[1,2]])
        pits=np.array([[0,3],[7,3], [9,9]])

    #Task 2
    if usetask == 2:
        # task 324 from ECE657A-W26-Asg1
        agentXY=[3,1] # Agent start position
        goalXY=[3,8] # Target position, terminal state
        wall_shape=np.array([[1,2],[1,3],[2,3],[4,3],[7,4],[3,6],[3,7],[2,7]])
        pits=np.array([[2,2],[3,4],[5,2],[0,5],[7,5],[0,6],[8,6],[0,7],[4,7],[2,8]])

    #Task 3
    if usetask == 3:
        # task 325 from ECE657A-W26-Asg1
        agentXY=[0,0] # Agent start position
        goalXY=[5,5] # Target position, terminal state
        wall_shape=np.array([[1,6],[2,5],[3, 4],[4,4],[5, 4],[6, 4],[7,5],[7,6], [5,8], [4,2],[5,2], [7,2],[8,3]])
        pits=np.array([[1,2],[4,6],[6,6],[6 ,8],[6,2], [7,0]])

    max_states = 100 - len(wall_shape) - len(pits) - 1

    experiments=[]

    # q-learning algorithm
    if (runalg1):
        name1 = "QLearning on T " + str(usetask)
        env1 = Maze(agentXY,goalXY,wall_shape, pits, showRender, name1)
        RL1 = rlalg1(actions=list(range(env1.n_actions)), **RLargs)
        data1={}
        env1.after(10, update(env1, RL1, data1, **EXPargs))
        env1.mainloop()
        experiments.append((name1, env1,RL1, data1))

    # sarsa algorithm
    if (runalg2):
        name2 = "SARSA on T " + str(usetask)
        env2 = Maze(agentXY,goalXY,wall_shape, pits, showRender, name2)
        RL2 = rlalg2(actions=list(range(env2.n_actions)), **RLargs)
        data2={}
        env2.after(10, update(env2, RL2, data2, **EXPargs))
        env2.mainloop()
        experiments.append((name2, env2,RL2, data2))

    # expected sarsa algorithm
    if (runalg4):
        name4 = "Exp-Sarsa on T " + str(usetask)
        env4 = Maze(agentXY,goalXY,wall_shape, pits, showRender, name4)
        RL4 = rlalg4(actions=list(range(env4.n_actions)), **RLargs)
        data4={}
        env4.after(10, update(env4, RL4, data4, **EXPargs))
        env4.mainloop()
        experiments.append((name4, env4,RL4, data4))

    # td(lambda) algorithm
    if (runalg5):
        name5 = "TD(Lambda) on T " + str(usetask)
        env5 = Maze(agentXY,goalXY,wall_shape, pits, showRender, name5)
        RL5 = rlalg5(actions=list(range(env5.n_actions)), **RLargs)
        data5={}
        env5.after(10, update(env5, RL5, data5, **EXPargs))
        env5.mainloop()
        experiments.append((name5, env5,RL5, data5))

    #SoWrong Algorithm - given to students as a demo
    if(runalg7):
        name7 = "SoWrongAlg on T " + str(usetask)
        env7 = Maze(agentXY,goalXY,wall_shape, pits, showRender, name7)
        RL7 = rlalg7(actions=list(range(env7.n_actions)), **RLargs)
        data7={}
        env7.after(10, update(env7, RL7, data7, **EXPargs))
        env7.mainloop()
        experiments.append((name7, env7,RL7, data7))
    

    print("All experiments complete")

    # print(f"Experiment Setup:\n - episodes:{episodes} VI_sweeps:{VI_sweeps} sim_speed:{sim_speed}") 
    print(f"Experiment Setup:\n - episodes:{episodes}\n - sim_speed:{sim_speed}\n") 

    for name, env, RL, data in experiments:
        print("[{}] : {} : max-rew={:.3f} med-{}={:.3f} var-{}={:.3f} max-episode-len={} {}".format(
            name, 
            RL.display_name, 
            np.max(data['global_reward']),
            window,
            np.median(data['global_reward'][-window:]), 
            window,
            np.var(data['global_reward'][-window:]),
            np.max(data['ep_length']),
            RLargsStr))
        if save_trajectories:
            print(f'Trajectories saved for {len(data["trajectories_per_episode"])} episodes')
            print(f'  Example trajectory for episode 0: {data["trajectories_per_episode"][0][:5]}')

    # get a timestamp for the start of the set of experiments
    timestr = time.strftime("d%Y%m%d_t%H%M%S")
    savedfileexp = 'exp'
    savedfiledesc = 'jointplot'
    savedfileloc = 'data/'
    saved_episodes_str = str(episodes) if episodes < 1000 else f'{int(episodes/1000)}k'
    savedfileexperdescrip = f'{savedfileexp}_T{usetask}_ep{saved_episodes_str}_{timestr}'
    # savedfileexperdescrip = f'{savedfileexp}_{timestr}_{i}_{env.MAZE_H}x{env.MAZE_W}'

    if(do_save_data or save_trajectories):
        for i, (name, env, RL, data) in enumerate(experiments):
            savedfilealgo = RL.display_name
            savedfileprefix =f'{savedfileloc}/{savedfileexperdescrip}_i{i}_{savedfilealgo}'
            if(do_save_data):
                saveData(name, env,RL,data, savedfileprefix)
            if(save_trajectories):
                #Plot the trajectories taken in each episode over top of the maze image
                plot_Trajectories(experiments[i], do_save_figure, f'{savedfileprefix}_traj')
                # plt.show() #only makes sense to show this as a popup window if you are only running one algorithm


    if(do_plot_rewards):
        #Simple plot of summed reward for each episode and algorithm, you can make more informative plots
        savedfileprefix =f'{savedfileloc}/{savedfileexperdescrip}_jointplot'
        plot_rewards(experiments, window, do_save_figure, savedfileprefix, logPlot=False)
        # plt.show()

