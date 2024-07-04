; The width of a lane: 3.5m (or 3.75m) ~ 2 * (patch-size)
; The width of a car: ~1.7m

extensions [ py ]

breed [cars car]

globals [
  selected-car          ; the currently selected car
  nb-cars-max           ; maximum number of car

  number-of-lanes       ; number of lanes for two-way
  lanes                 ; a list of the y coordinates of different lanes
  car-lanes
  car-lanes-left        ; a list of the y coordinates of different lanes for cars (from left to right)
  car-lanes-right       ; a list of the y coordinates of different lanes for cars (from right to left)
  ;rescue-lanes         ; a list of the y coordinates of rescue lanes

  speed-max
  speed-min
  speed-sudden-stop     ; the speed that a car has when its car ahead is damaged
  speed-ratio           ; the ratio to multiply with speed

  collision-distance    ; the distance that can be considered as collission
  nb-collisions         ; number of collisions

  damaged-color                 ; the color for damaged car
  damaged-inlane-duration-max   ; the maximum duration for a damaged car stand on running lanes
  damaged-inrescue-duration-max ; the maximum duration for a damaged car stand on rescue lanes
  damaged-nb-cars-inlane        ; the maximum number of damaged cars who stand on running lanes

  change-lane-step      ; the value that we forward a car when it changes lane
  move-rescue-lane-step ; the value that we forward a damaged car to the rescue lanes

  prob-damaged-car      ; the rate at which a car is damaged when traveling on a road
  prob-add-car          ; the probability that we add a car
  prob-remove-car       ; the probability that we remove a damaged car

  observation-max       ; the maximum distance that a car can observe
  observation-distance  ; the distance that we use to observe blocking cars
  observation-angle     ; the angle that we use to observe blocking cars

  ; parameters for rl algorithms
  state-env
  ac-update-steps
]

; define attributes for car agent
cars-own [
  speed                       ; the current speed of the car
  speed-top                   ; the maximum speed of the car (different for all cars)
  target-lane                 ; the desired lane of the car
  patience                    ; the driver's current level of patience
  damaged-inlane-duration     ; the duration that a damaged car is in running lanes
  damaged-inrescue-duration   ; the duration that a damaged car is in rescue lanes

  ; attributes for rl
  reward
  state
  action        ; 0 = decelerate, 1 = stay same, 2 = accelerate, 3 = change lane
  next-state
]

to setup
  clear-all

  set number-of-lanes (number-of-lanes-way * 2 + 3)
  set nb-cars-max round (number-of-lanes-way * 1.5 * world-width * 0.25)

  set speed-max 1.0 ; ~ 120 km/h
  set speed-min 0.0 ;
  set speed-sudden-stop 0.05
  set speed-ratio 0.35 ; need to recalculate

  set collision-distance  1.0

  set damaged-color orange
  set damaged-inlane-duration-max 1800 ; 30 minutes
  set damaged-inrescue-duration-max 2700 ; 45 minutes
  set damaged-nb-cars-inlane 0

  set change-lane-step 0.08
  set move-rescue-lane-step change-lane-step * 0.1

  set prob-damaged-car 0.00002
  set prob-add-car 0.005
  set prob-remove-car 0.00005

  set observation-max world-width / 2
  set observation-distance 2.5
  set observation-angle 60 ;45

  set ac-update-steps 2000

  draw-road
  create-or-remove-cars

  set selected-car one-of cars
  ask selected-car [ set color pink ]


  ifelse ((driving-policy = "Greedy") or (driving-policy = "Greedy-CL")) [
    ; use Greedy strategies
  ][
    ; use RL strategies
    py:setup py:python
    py:run "import numpy as np"

    py:set "action_size" 4   ; 0 = decelerate, 1 = stay same, 2 = accelerate, 3 = change lane
    py:set "gamma" discount-factor
    py:set "alpha" learning-rate

    if (driving-policy = "SARSA") or (driving-policy = "Q-Learning") or (driving-policy = "Double Q-Learning") [
      setup-tabular-algos
    ]

    if (driving-policy = "Deep Q-Learning") or (driving-policy = "Deep Q-Network") or (driving-policy = "Double Deep Q-Learning") [
      setup-approximate-algos
    ]

    if (driving-policy = "Naive Actor-Critic") or (driving-policy = "Advantage Actor-Critic") or (driving-policy = "Soft Actor-Critic") [
      setup-actor-critic-algos
    ]

    if (driving-policy = "Proximal Policy Optimization")  [
      setup-ppo
    ]
  ]

  reset-ticks
end

to go
  if ticks >= simulation-time [stop]

  if dynamic-situation? [
    if (random-float 1 <= prob-add-car) and (number-of-cars < nb-cars-max) [
      set number-of-cars (number-of-cars + 1) ; add cars
    ]
  ]

  set nb-collisions 0

  create-or-remove-cars

  if ( (driving-policy = "Greedy") or (driving-policy = "Greedy-CL") ) [
    ;ask cars with [speed > 0] [ move-forward-greedy ]
    ask cars [ move-forward-greedy ]
  ]

  if (driving-policy = "SARSA") or (driving-policy = "Q-Learning") [
    go-sarsa-qlearning
  ]

  if (driving-policy = "Double Q-Learning") [
    go-double-ql
  ]

  if (driving-policy = "Deep Q-Learning") or (driving-policy = "Deep Q-Network") or (driving-policy = "Double Deep Q-Learning") [
    go-approximate-algos
  ]

  if (driving-policy = "Naive Actor-Critic") [
    go-nac
  ]

  if (driving-policy = "Advantage Actor-Critic") [
    ifelse multiple-workers?[
      go-a2c-with-workers
    ][
      go-a2c-without-workers
    ]
  ]

  if (driving-policy = "Soft Actor-Critic") [
    go-sac
  ]

  if (driving-policy = "Proximal Policy Optimization")  [
    go-ppo
  ]

  tick
end

;; save the simulation to file
to save
end

;; load the simulation from file
to load
end

;; setup function for tabular algorithms (i.e., Q-Learning, Double Q-Learning, SARSA)
to setup-tabular-algos
  set state-env (list
    [-> patience]
    [-> round (100 * speed)]
    [-> round (get-distance-to-car get-car-ahead)]
    [-> round (100 * get-speed-to-car get-car-ahead)])

  let state-size  (max-patience + 1) * 101 * world-width * 101; speed: 0 -> 100 -- distance: 0 - world-width -- speed-car-ahead: 0 -> 100
  py:set "state_size" state-size

  if (driving-policy = "SARSA") or (driving-policy = "Q-Learning") [
    py:run "Q = np.zeros([state_size, action_size])"  ; initialize Q-table values to 0
  ]

  if driving-policy = "Double Q-Learning" [
    py:run "QA = np.zeros([state_size, action_size])"  ; initialize QA-table values to 0
    py:run "QB = np.zeros([state_size, action_size])"  ; initialize QB-table values to 0
  ]

end

;; setup for approximate algorithms (i.e., Deep Q-Learning, Deep Q-Network, ...)
;; in fact, DQL is the a version of DQN without Target Network
to setup-approximate-algos
  set state-env (list
    [-> speed]
    [-> get-distance-to-car get-car-ahead]
    [-> get-speed-to-car get-car-ahead]
    [-> get-distance-to-car get-car-behind]
    [-> get-speed-to-car get-car-behind])

  if input-exp? [
    set state-env lput [-> get-epsilon ] state-env
  ]

  if input-time? [
    set state-env lput [-> ticks] state-env
  ]

  py:set "input_state_size" length state-env
  ;print length state-env
  py:set "hidden_layer_size" 36
  py:set "memory_size" memory-size
  py:set "batch_size" batch-size

  (py:run
    "from tensorflow import keras"
    "from keras import layers"
    ;; Q Network

    "Q_network = keras.Sequential()"
    "Q_network.add(layers.Dense(hidden_layer_size, input_shape=(input_state_size,), activation='relu'))"
    "Q_network.add(layers.Dense(hidden_layer_size, activation='relu'))"
    "Q_network.add(layers.Dense(hidden_layer_size, activation='relu'))"
    "Q_network.add(layers.Dense(action_size))"
    "optimizer = keras.optimizers.Adam(learning_rate=alpha)"
    "Q_network.compile(optimizer, 'mse')"
    ;"Q_network.summary()"

    ;; shared replay memory between all cars
    "memory = []")

  if (driving-policy = "Deep Q-Network") or (driving-policy = "Double Deep Q-Learning") [
    py:set "update_steps" 1000    ;; hyperparameter to update target (before: c = 5)
    py:set "step" 0

    (py:run "Q_hat_network = keras.models.clone_model(Q_network)")
  ]

end

;; setup for actor-critic algorithms (Naive AC, A2C without multiple workers, A2C with multiple workers, Soft Actor-Critic)
;; use neural networks: for Actor and for Critic
;; Actor network is used to select action
;; Critic network is used to calculate state value
;; A policy function (or policy) returns a probability distribution over actions that the agent can take based on the given state
to setup-actor-critic-algos
  set state-env (list
    [-> speed]
    [-> get-distance-to-car get-car-ahead]
    [-> get-speed-to-car get-car-ahead]
    [-> get-distance-to-car get-car-behind]
    [-> get-speed-to-car get-car-behind])

  if input-exp? [
    set state-env lput [-> get-epsilon ] state-env
  ]

  if input-time? [
    set state-env lput [-> ticks] state-env
  ]

  py:set "input_state_size" length state-env
  py:set "hidden_layer_size" 36

  (py:run
    "import tensorflow as tf"
    "import tensorflow_probability as tfp"
    "from tensorflow import keras"
    "from keras import layers"
    "import tensorflow.keras.losses as kls"

    ;; Actor network
    "Actor_network = keras.Sequential()"
    "Actor_network.add(layers.Dense(hidden_layer_size, input_shape=(input_state_size,), activation='relu'))"
    "Actor_network.add(layers.Dense(hidden_layer_size, activation='relu'))"
    "Actor_network.add(layers.Dense(hidden_layer_size, activation='relu'))"
    "Actor_network.add(layers.Dense(action_size, activation='softmax'))"
    "a_optimizer = keras.optimizers.Adam(learning_rate=alpha)"
    "Actor_network.compile(a_optimizer, 'mse')"
    ;"Actor_network.summary()"

    ;; Critic network
    "Critic_network = keras.Sequential()"
    "Critic_network.add(layers.Dense(hidden_layer_size, input_shape=(input_state_size,), activation='relu'))"
    "Critic_network.add(layers.Dense(hidden_layer_size, activation='relu'))"
    "Critic_network.add(layers.Dense(hidden_layer_size, activation='relu'))"
    "Critic_network.add(layers.Dense(1))"
    "c_optimizer = keras.optimizers.Adam(learning_rate=alpha)"
    "Critic_network.compile(c_optimizer, 'mse')"
    ;"Critic_network.summary()"
  )

  if driving-policy = "Advantage Actor-Critic" [
    py:set "update_steps" ac-update-steps
    py:set "update_offset" (round (ac-update-steps / number-of-cars) )
    py:set "step" 0
    py:set "nb_cars" ((max [ who ] of cars) + 1)

    ifelse multiple-workers?[ ; with multi-workers A2C
      (py:run
        "from multiprocessing import Process, Queue, Barrier, Lock"
        "barrier = Barrier(number_workers)"
        "s_queue = Queue()"
        "a_queue = Queue()"
        "r_queue = Queue()"
        "lock = Lock()"
        "processes = []")
    ][ ; without multi-workers A2C
      ; 3 matrices to store state, action, reward for each car (review 22/12/2022: should it be shared memories for all car ???)
      (py:run
        "mem_states = [[] for i in range(nb_cars)]"
        "mem_rewards = [[] for i in range(nb_cars)]"
        "mem_actions = [[] for i in range(nb_cars)]"
        "mem_returns = [[] for i in range(nb_cars)]")
    ]
  ]

  if driving-policy = "Soft Actor-Critic" [
    py:set "memory_size" memory-size
    py:set "batch_size" batch-size
    py:set "nb_cars" ((max [ who ] of cars) + 1)

    (py:run
      ;; Critic2 network
      "Critic2_network = keras.Sequential()"
      "Critic2_network.add(layers.Dense(hidden_layer_size, input_shape=(input_state_size,), activation='relu'))"
      "Critic2_network.add(layers.Dense(hidden_layer_size, activation='relu'))"
      "Critic2_network.add(layers.Dense(hidden_layer_size, activation='relu'))"
      "Critic2_network.add(layers.Dense(1))"
      "c2_optimizer = keras.optimizers.Adam(learning_rate=alpha)"
      "Critic2_network.compile(c2_optimizer, 'mse')"

      ;; Value network
      "Value_network = keras.Sequential()"
      "Value_network.add(layers.Dense(hidden_layer_size, input_shape=(input_state_size,), activation='relu'))"
      "Value_network.add(layers.Dense(hidden_layer_size, activation='relu'))"
      "Value_network.add(layers.Dense(hidden_layer_size, activation='relu'))"
      "Value_network.add(layers.Dense(1))"
      "v_optimizer = keras.optimizers.Adam(learning_rate=alpha)"
      "Value_network.compile(v_optimizer, 'mse')"

      ;; Target Value network
      "Target_Value_network = keras.Sequential()"
      "Target_Value_network.add(layers.Dense(hidden_layer_size, input_shape=(input_state_size,), activation='relu'))"
      "Target_Value_network.add(layers.Dense(hidden_layer_size, activation='relu'))"
      "Target_Value_network.add(layers.Dense(hidden_layer_size, activation='relu'))"
      "Target_Value_network.add(layers.Dense(1))"
      "Target_Value_network.compile(v_optimizer, 'mse')"

      ;; Replay buffer
      ;"state_memory = [np.zeros((memory_size,), dtype=np.float32) for i in range(nb_cars)]"
      ;"action_memory = [np.zeros((memory_size, action_size), dtype=np.float32) for i in range(nb_cars)]"
      ;"reward_memory = [np.zeros((memory_size,), dtype=np.float32) for i in range(nb_cars)]"
      ;"next_state_memory = [np.zeros((memory_size,), dtype=np.float32) for i in range(nb_cars)]"

      ;; shared replay memory between all cars
      "memory = []"
      "ep_reward = []"
      "total_avgr = []"
      "total_reward = []"
    )
  ]

end

;; setup for actor-critic algorithms (Naive AC, A2C without multiple workers, A2C with multiple workers, Soft Actor-Critic)
;; use neural networks: for Actor and for Critic
;; Actor network is used to select action
;; Critic network is used to calculate state value
;; A policy function (or policy) returns a probability distribution over actions that the agent can take based on the given state
to setup-ppo
  set state-env (list
    [-> speed]
    [-> get-distance-to-car get-car-ahead]
    [-> get-speed-to-car get-car-ahead]
    [-> get-distance-to-car get-car-behind]
    [-> get-speed-to-car get-car-behind])

  if input-exp? [
    set state-env lput [-> get-epsilon ] state-env
  ]

  if input-time? [
    set state-env lput [-> ticks] state-env
  ]

  py:set "input_state_size" length state-env
  py:set "hidden_layer_size" 36

  (py:run
    "import tensorflow as tf"
    "import tensorflow_probability as tfp"
    "from tensorflow import keras"
    "from keras import layers"
    "import tensorflow.keras.losses as kls"

    ;; Actor network
    "Actor_network = keras.Sequential()"
    "Actor_network.add(layers.Dense(hidden_layer_size, input_shape=(input_state_size,), activation='relu'))"
    "Actor_network.add(layers.Dense(hidden_layer_size, activation='relu'))"
    "Actor_network.add(layers.Dense(hidden_layer_size, activation='relu'))"
    "Actor_network.add(layers.Dense(action_size, activation='softmax'))"
    "a_optimizer = keras.optimizers.Adam(learning_rate=alpha)"
    "Actor_network.compile(a_optimizer, 'mse')"
    ;"Actor_network.summary()"

    ;; Critic network
    "Critic_network = keras.Sequential()"
    "Critic_network.add(layers.Dense(hidden_layer_size, input_shape=(input_state_size,), activation='relu'))"
    "Critic_network.add(layers.Dense(hidden_layer_size, activation='relu'))"
    "Critic_network.add(layers.Dense(hidden_layer_size, activation='relu'))"
    "Critic_network.add(layers.Dense(1))"
    "c_optimizer = keras.optimizers.Adam(learning_rate=alpha)"
    "Critic_network.compile(c_optimizer, 'mse')"
    ;"Critic_network.summary()"
  )

  if driving-policy = "Advantage Actor-Critic" [
    py:set "update_steps" ac-update-steps
    py:set "update_offset" (round (ac-update-steps / number-of-cars) )
    py:set "step" 0
    py:set "nb_cars" ((max [ who ] of cars) + 1)

    ; 3 matrices to store state, action, reward
    (py:run
      "mem_states = [[] for i in range(nb_cars)]"
      "mem_rewards = [[] for i in range(nb_cars)]"
      "mem_actions = [[] for i in range(nb_cars)]"
      "mem_returns = [[] for i in range(nb_cars)]")
  ]

  if driving-policy = "Soft Actor-Critic" [
    py:set "memory_size" memory-size
    py:set "batch_size" batch-size

    (py:run
      ;; Critic2 network
      "Critic2_network = keras.Sequential()"
      "Critic2_network.add(layers.Dense(hidden_layer_size, input_shape=(input_state_size,), activation='relu'))"
      "Critic2_network.add(layers.Dense(hidden_layer_size, activation='relu'))"
      "Critic2_network.add(layers.Dense(hidden_layer_size, activation='relu'))"
      "Critic2_network.add(layers.Dense(1))"
      "c2_optimizer = keras.optimizers.Adam(learning_rate=alpha)"
      "Critic2_network.compile(c2_optimizer, 'mse')"

      ;; Replay memory
      "memory = []"
    )
  ]

end

;; go function for Double Q-Learning algorithm
;; 18/01/2023: modify this to use only one agent to update Q table ???
to go-sarsa-qlearning
  ask cars [
    set state map runresult state-env ; get current state
    let state-int convert-state-int state
    py:set "state" state-int

    ; choose an action corresponding current state, using e-greedy
    ifelse (0.05 + random-float 1) < get-epsilon [
      set action random 4 ; explore the random / new action
      while [action = 3 and patience > 0] [set action random 4]
    ][
      set action py:runresult "np.argmax(Q[state,:])" ; find the best action for this state
    ]
    py:set "action" action

    ; perform choosen action to get next reward and next state
    move-forward-rl

    py:set "reward" reward

    set next-state map runresult state-env
    let next-state-int convert-state-int next-state
    py:set "next_state" next-state-int

    ; update Q-table
    if driving-policy = "SARSA" [
      ; using e-greedy to choose next action
      let next_action py:runresult "np.argmax(Q[next_state,:])" ; find the best action for next state
      if (0.05 + random-float 1) < get-epsilon [
        set next_action random 4 ; explore the random / new action
        while [next_action = 3 and patience > 0] [set next_action random 4]
      ]
      py:set "next_action" next_action
      py:run "Q[state,action] += alpha * (reward + gamma * Q[next_state,next_action] - Q[state,action])"
    ]

    if driving-policy = "Q-Learning" [
      py:run "Q[state,action] += alpha * (reward + gamma * np.max(Q[next_state,:]) - Q[state,action])"
    ]
  ]
end

;; go function for Double Q-Learning algorithm
;; 18/01/2023: modify this to use only one agent to update Q table ???
to go-double-ql
  ask cars [
    set state map runresult state-env ; get current state
    let state-int convert-state-int state
    py:set "state" state-int

    ; choose an action corresponding current state, using e-greedy
    ifelse (0.05 + random-float 1) < get-epsilon [
      set action random 4 ; explore the random / new action
      while [action = 3 and patience > 0] [set action random 4]
    ][
      set action py:runresult "np.argmax( (QA[state,:] + QB[state,:])/2 )" ; find the best action for this state
    ]
    py:set "action" action

    ; perform choosen action to get next reward and next state
    move-forward-rl

    py:set "reward" reward

    set next-state map runresult state-env
    let next-state-int convert-state-int next-state
    py:set "next_state" next-state-int

    ; update QA, QB tables
    ifelse (random-float 1) < 0.5 [  ; update QA
      py:run "next_action = np.argmax(QA[next_state,:])"
      py:run "QA[state,action] += alpha * (reward + gamma * np.max(QB[next_state,next_action]) - QA[state,action])"
    ][ ; update QB
      py:run "next_action = np.argmax(QB[next_state,:])"
      py:run "QB[state,action] += alpha * (reward + gamma * np.max(QA[next_state,next_action]) - QB[state,action])"
    ]
  ]
end

;; go for Deep Q-Learning, Deep Q-Network and Double Deep Q-Learning
to go-approximate-algos
  ; get current state for all cars
  ask cars [
    set state map runresult state-env
  ]
  let car-list sort cars
  py:set "states" map [ t -> [ state ] of t ] car-list

  ; agents choose actions, using e-greedy
  let actions py:runresult "np.argmax(Q_network.predict(np.array(states)), axis = 1)"
  (foreach car-list actions [ [t a] ->
    ask t [
      ifelse (0.05 + random-float 1) < get-epsilon [
        set action random 4
        while [action = 3 and patience > 0] [set action random 4]
      ] [
        set action a
      ]
    ]
  ])

  ; perform chosen actions, i.e. forward cars and get next state
  ask cars [
    move-forward-rl
    set next-state map runresult state-env
  ]

  ; save experience to share memory in order to train Q-network
  let data [ (list state action reward next-state) ] of cars
  py:set "new_experience" data
  (py:run
    "memory.extend(new_experience)"
    "if len(memory) > memory_size:"
    "    memory = memory[-memory_size:]"
    "sample_ix = np.random.randint(len(memory), size = batch_size)"
    "states = np.array([memory[i][0] for i in sample_ix])"
    "actions = np.array([memory[i][1] for i in sample_ix])"
    "rewards = np.array([memory[i][2] for i in sample_ix])"
    "next_states = np.array([memory[i][3] for i in sample_ix])"
    "targets = Q_network.predict(states)"
  )

  if driving-policy = "Deep Q-Learning" [
    (py:run
      "q_values = Q_network.predict(next_states)"
      "targets[np.arange(targets.shape[0]), actions] = rewards + gamma*np.max(q_values, axis = 1)") ; axis = 1 means to find max value along rows
  ]

  if driving-policy = "Deep Q-Network" [
    (py:run
      "q_values = Q_hat_network.predict(next_states)"
      "targets[np.arange(targets.shape[0]), actions] = rewards + gamma*np.max(q_values, axis = 1)"); ; axis = 1 means to find max value along rows
  ]

  if driving-policy = "Double Deep Q-Learning"  [
    (py:run
      "print(next_states)"
      ;"next_actions = np.argmax(Q_network.predict(next_states), axis = 1)"
      )
      ;"print(next_actions)"
      ;"q_values = Q_hat_network.predict(next_states)"
      ;"q_actions_values = [q_values[i] for i in next_actions]"
      ;"targets[np.arange(targets.shape[0]), actions] = rewards + gamma*q_actions_values")
  ]

  py:run "Q_network.train_on_batch(states, targets)"

  if (driving-policy = "Deep Q-Network") or (driving-policy = "Double Deep Q-Learning") [
    ; update periodically target network
    (py:run
      "step = step + 1"
      "if (step % update_steps == 0): Q_hat_network = keras.models.clone_model(Q_network)")
  ]
end

;; go for Naive Actor-Critic
;; Naive AC: update neural networks on each timestamp
to go-nac
  ; get current states
  ask cars [
    set state map runresult state-env
  ]
  let car-list sort cars
  py:set "states" map [ t -> [ state ] of t ] car-list
  ;py:run "print(np.array(states).shape)"

  ; using Actor network to choose actions, using e-greedy
  let actions py:runresult "np.argmax(Actor_network.predict(np.array(states)), axis = 1)"
  (foreach car-list actions [ [t a] ->
    ask t [
      ifelse (0.05 + random-float 1) <= get-epsilon [
        set action random 4
        while [action = 3 and patience > 0] [set action random 4]
      ] [
        set action a
      ]
    ]
  ])

  ; perform chosen actions, i.e. forward cars and get next state
  ask cars [
    move-forward-rl
    set next-state map runresult state-env
  ]

  py:set "actions" map [ t -> [ action ] of t ] car-list
  py:set "rewards" map [ t -> [ reward ] of t ] car-list
  py:set "next_states" map [ t -> [ next-state ] of t ] car-list

  ; calculate losses of Actor and Critic
  ; - Actor loss is negative of Log probability of action taken multiplied by temporal difference used in Q-learning
  ; - Critic loss is square of the temporal difference
  (py:run
    "with tf.GradientTape() as tape1, tf.GradientTape() as tape2:"
    "   policies = Actor_network(np.array(states))"
    "   values = Critic_network(np.array(states))"
    "   values_next = Critic_network(np.array(next_states))"
    "   tds = rewards + gamma*values_next - values"
    "   dists = tfp.distributions.Categorical(probs=policies, dtype=tf.float32)"
    "   log_probs = dists.log_prob(actions)"
    "   actor_losses = -log_probs*tds"
    "   critic_losses = tds**2"
    "grads1 = tape1.gradient(actor_losses, Actor_network.trainable_variables)"
    "grads2 = tape2.gradient(critic_losses, Critic_network.trainable_variables)"
    "a_optimizer.apply_gradients(zip(grads1, Actor_network.trainable_variables))"
    "c_optimizer.apply_gradients(zip(grads2, Critic_network.trainable_variables))")
end

;; go for Soft Actor-Critic
;; off-policy maximum entroy deep RL
to go-sac
  ; get current state for all cars
  ask cars [
    set state map runresult state-env
  ]
  let car-list sort cars
  py:set "states" map [ t -> [ state ] of t ] car-list

  ; using Actor network to choose actions, using e-greedy
  let actions py:runresult "np.argmax(Actor_network.predict(np.array(states)), axis = 1)"
  (foreach car-list actions [ [t a] ->
    ask t [
      ifelse (0.05 + random-float 1) <= get-epsilon [
        set action random 4
        while [action = 3 and patience > 0] [set action random 4]
      ] [
        set action a
      ]
    ]
  ])

  ; perform chosen actions, i.e. forward cars and get next state
  ask cars [
    move-forward-rl
    set next-state map runresult state-env
  ]

  ;py:set "actions" map [ t -> [ action ] of t ] car-list
  ;py:set "rewards" map [ t -> [ reward ] of t ] car-list
  ;py:set "next_states" map [ t -> [ next-state ] of t ] car-list

  ; save experience to the share replay memory
  let data [ (list state action reward next-state) ] of cars
  py:set "new_experience" data
  (py:run
    "memory.extend(new_experience)"
    "if len(memory) > memory_size:"
    "    memory = memory[-memory_size:]"
    "sample_ix = np.random.randint(len(memory), size = batch_size)"
    "states = np.array([memory[i][0] for i in sample_ix])"
    "actions = np.array([memory[i][1] for i in sample_ix])"
    "rewards = np.array([memory[i][2] for i in sample_ix])"
    "next_states = np.array([memory[i][3] for i in sample_ix])"

    "states = tf.convert_to_tensor(states, dtype= tf.float32)"
    "next_states = tf.convert_to_tensor(next_states, dtype= tf.float32)"
    "rewards = tf.convert_to_tensor(rewards, dtype= tf.float32)"
    "actions = tf.convert_to_tensor(actions, dtype= tf.float32)"

    "with tf.GradientTape() as tape1, tf.GradientTape() as tape2, tf.GradientTape() as tape3, tf.GradientTape() as tape4:"
    "   value = tf.squeeze(self.value_net(states))"
    "   policies = Actor_network(np.array(states))"
    "   values = Critic_network(np.array(states))"
    "   values_next = Critic_network(np.array(next_states))"
    "   tds = rewards + gamma*values_next - values"
    "   dists = tfp.distributions.Categorical(probs=policies, dtype=tf.float32)"
    "   log_probs = dists.log_prob(actions)"
    "   actor_losses = -log_probs*tds"
    "   critic_losses = tds**2"
    "grads1 = tape1.gradient(actor_losses, Actor_network.trainable_variables)"
    "grads2 = tape2.gradient(critic_losses, Critic_network.trainable_variables)"
    "a_optimizer.apply_gradients(zip(grads1, Actor_network.trainable_variables))"
    "c_optimizer.apply_gradients(zip(grads2, Critic_network.trainable_variables))")
end

;; go for A2C WITHOUT multiple workers
;; update neural networks after every n time steps
to go-a2c-without-workers
  ; get current states
  ask cars [
    set state map runresult state-env
  ]
  let car-list sort cars
  py:set "states" map [ t -> [ state ] of t ] car-list

  ; using Actor network to choose actions, using e-greedy
  let actions py:runresult "np.argmax(Actor_network.predict(np.array(states)), axis = 1)"
  (foreach car-list actions [ [t a] ->
    ask t [
      ifelse (0.05 + random-float 1) <= get-epsilon [
        set action random 4
        while [action = 3 and patience > 0] [set action random 4]
      ] [
        set action a
      ]
    ]
  ])

  ; perform chosen actions, i.e. forward cars and get next state
  ask cars [
    move-forward-rl
    set next-state map runresult state-env
  ]

  ; stores current state, action, reward for each car
  py:run "step = step + 1"
  ;; 18/01/2023: modify this to update 1 time by only one agent ???
  ask cars [
    py:set "id" who
    py:set "state" state
    py:run "mem_states[id].append(state)"
    py:set "action" action
    py:run "mem_actions[id].append(action)"
    py:set "reward" reward
    py:run "mem_rewards[id].append(reward)"

    (py:run
      "if ((step % update_steps) == (id * update_offset)):"
      "   mem_returns[id] = []"
      "   sum_reward = 0"
      "   mem_rewards[id].reverse()"
      "   for r in mem_rewards[id]:"
      "      sum_reward = r + gamma*sum_reward"
      "      mem_returns[id].append(sum_reward)"
      "   mem_returns[id].reverse()"
      "   mem_states[id] = np.array(mem_states[id], dtype=np.float32)"
      "   mem_actions[id] = np.array(mem_actions[id], dtype=np.float32)"
      "   mem_returns[id] = np.array(mem_returns[id], dtype=np.float32)"
      "   mem_returns[id] = tf.reshape(mem_returns[id], (len(mem_returns[id]),))"
      ;; learn function and loss
      "   with tf.GradientTape() as tape1, tf.GradientTape() as tape2:"
      "      policy = Actor_network(mem_states[id], training=True)"
      "      value = Critic_network(mem_states[id], training=True)"
      "      value = tf.reshape(value, (len(value),))"
      "      td = tf.math.subtract(mem_returns[id], value)"
      "      probability = []"
      "      log_probability = []"
      "      for pb, a in zip(policy, mem_actions[id]):"
      "         dist = tfp.distributions.Categorical(probs=pb, dtype=tf.float32)"
      "         log_prob = dist.log_prob(a)"
      "         prob = dist.prob(a)"
      "         probability.append(prob)"
      "         log_probability.append(log_prob)"
      "      p_loss = []"
      "      e_loss = []"
      "      td = td.numpy()"
      "      for pb, t, lpb in zip(probability, td, log_probability):"
      "         t =  tf.constant(t)"
      "         policy_loss = tf.math.multiply(lpb, t)"
      "         entropy_loss = tf.math.negative(tf.math.multiply(pb,lpb))"
      "         p_loss.append(policy_loss)"
      "         e_loss.append(entropy_loss)"
      "      p_loss = tf.stack(p_loss)"
      "      e_loss = tf.stack(e_loss)"
      "      p_loss = tf.reduce_mean(p_loss)"
      "      e_loss = tf.reduce_mean(e_loss)"
      "      actor_loss = -p_loss - 0.0001*e_loss"
      "      critic_loss = 0.5*kls.mean_squared_error(mem_returns[id], value)"
      "   grads1 = tape1.gradient(actor_loss, Actor_network.trainable_variables)"
      "   grads2 = tape2.gradient(critic_loss, Critic_network.trainable_variables)"
      "   a_optimizer.apply_gradients(zip(grads1, Actor_network.trainable_variables))"
      "   c_optimizer.apply_gradients(zip(grads2, Critic_network.trainable_variables))"
      ;; reset memory: states, actions, rewards
      "   mem_states[id] = []"
      "   mem_actions[id] = []"
      "   mem_rewards[id] = []")
  ]
end

;; go for A2C WITH multiple workers
to go-a2c-with-workers
end

;; go for Proximal Policy Optimization
to go-ppo
  ; get current states
  ask cars [
    set state map runresult state-env
  ]
  let car-list sort cars
  py:set "states" map [ t -> [ state ] of t ] car-list
  ;py:run "print(np.array(states).shape)"

  ; using Actor network to choose actions, using e-greedy
  let actions py:runresult "np.argmax(Actor_network.predict(np.array(states)), axis = 1)"
  (foreach car-list actions [ [t a] ->
    ask t [
      ifelse (0.05 + random-float 1) <= get-epsilon [
        set action random 4
        while [action = 3 and patience > 0] [set action random 4]
      ] [
        set action a
      ]
    ]
  ])

  ; perform chosen actions, i.e. forward cars and get next state
  ask cars [
    move-forward-rl
    set next-state map runresult state-env
  ]

  py:set "actions" map [ t -> [ action ] of t ] car-list
  py:set "rewards" map [ t -> [ reward ] of t ] car-list
  py:set "next_states" map [ t -> [ next-state ] of t ] car-list

  ; calculate losses of Actor and Critic
  ; - Actor loss is negative of Log probability of action taken multiplied by temporal difference used in Q-learning
  ; - Critic loss is square of the temporal difference
  (py:run
    "with tf.GradientTape() as tape1, tf.GradientTape() as tape2:"
    "   policies = Actor_network(np.array(states))"
    "   values = Critic_network(np.array(states))"
    "   values_next = Critic_network(np.array(next_states))"
    "   tds = rewards + gamma*values_next - values"
    "   dists = tfp.distributions.Categorical(probs=policies, dtype=tf.float32)"
    "   log_probs = dists.log_prob(actions)"
    "   actor_losses = -log_probs*tds"
    "   critic_losses = tds**2"
    "grads1 = tape1.gradient(actor_losses, Actor_network.trainable_variables)"
    "grads2 = tape2.gradient(critic_losses, Critic_network.trainable_variables)"
    "a_optimizer.apply_gradients(zip(grads1, Actor_network.trainable_variables))"
    "c_optimizer.apply_gradients(zip(grads2, Critic_network.trainable_variables))")
end

;; forward car according greedy algorithm
to move-forward-greedy ; car procedure
  ifelse speed > 0 [ ; for car with speed > 0
    ifelse (prob-damaged-car <= random-float 1) [
      if (pycor < 0) [ set heading 90 ]
      if (pycor > 0) [ set heading 270 ]

      set color get-car-default-color self

      speed-up-car ; tentatively speed up, but might have to slow down

      handle-blocking-cars

      forward speed * speed-ratio

      if patience <= 0  [ choose-new-lane ]  ; want to change lane
      if ycor != target-lane [ move-to-target-lane ]
    ][
      if (damaged-nb-cars-inlane < max-damaged-cars) [ ; this car will stop
        set speed 0
        set color damaged-color
        set damaged-inlane-duration 1
        set damaged-nb-cars-inlane (damaged-nb-cars-inlane + 1)
      ]
    ]
  ][ ; for car with speed = 0
    handle-damaged-car
  ]
end

;; forward car according rl algorithms
to move-forward-rl ; car procedure
  ifelse speed > 0 [
    ifelse (prob-damaged-car <= random-float 1) [
      ;ifelse (member? pycor car-lanes-left) [ set heading 90 ][ set heading 270 ]
      if (pycor < 0) [ set heading 90 ]
      if (pycor > 0) [ set heading 270 ]

      if action = 0 [ speed-up-car set color green ]   ; 0 = accelerate, green
      if action = 1 [ set color yellow ]               ; 1 = stay same, yellow
      if action = 2 [ slow-down-car set color red]     ; 2 = decelerate, red
      if action = 3 [                                  ; 3 = change lane, blue
        choose-new-lane
        set color blue
        if ycor != target-lane [
          move-to-target-lane
        ]
      ]

      handle-blocking-cars

      set reward (log (speed + (max-patience - patience) / max-patience + 1e-8) 2) ; reward can be positive or negative

      forward speed * speed-ratio
    ][
      if (damaged-nb-cars-inlane < max-damaged-cars) [ ; this car will stop
        set speed 0
        set color damaged-color
        set damaged-inlane-duration 1
        set damaged-nb-cars-inlane (damaged-nb-cars-inlane + 1)
      ]
    ]
  ][
    ; for car with speed = 0
    handle-damaged-car
  ]
end


; handle blocking cars
to handle-blocking-cars ; car procedure
  ;let blocking-y-cars other cars in-cone (observation-distance + speed) observation-angle with [ get-x-distance <= observation-distance ]
  ;set observation-distance get-safe-distance
  let blocking-cars other cars in-cone observation-distance observation-angle
  let blocking-car-nearest min-one-of blocking-cars [ distance myself ]
  if blocking-car-nearest != nobody [
    ifelse [ speed ] of blocking-car-nearest > 0 [
      ;while [speed >= ([ speed ] of blocking-y-car)] [slow-down-car]
      set speed [ speed ] of blocking-car-nearest ; match the speed of the car ahead of you and then slow down so you are driving a bit slower than that car.
      slow-down-car
    ][  ; if blocking-car-nearest is a damaged car, with speed = 0
      while [speed > speed-sudden-stop] [slow-down-car]
      ; set speed speed-sudden-stop
      while [ycor = target-lane] [
        choose-new-lane
        if ycor != target-lane [
          move-to-target-lane
        ]
      ]
    ]
  ]
end

; handle the damaged cars, with speed = 0
to handle-damaged-car ; car procedure
  if (dynamic-situation?) [
    ifelse ((ycor > 1 - number-of-lanes) and (ycor < number-of-lanes - 1))[
      set damaged-inlane-duration (damaged-inlane-duration + 1)
      if damaged-inlane-duration >= damaged-inlane-duration-max [ ; move car to rescue lane
        if (ycor < 0) [ set heading 180 ]
        if (ycor > 0) [ set heading 0 ]

        if not any? cars-on patch-ahead 1 [
          forward move-rescue-lane-step
        ]

        if ( (ycor <= 1 - number-of-lanes) or (ycor >= number-of-lanes - 1) ) [
          set damaged-inlane-duration 0
          set damaged-inrescue-duration 1
          set damaged-nb-cars-inlane (damaged-nb-cars-inlane - 1)
          ifelse (ycor <= 1 - number-of-lanes) [set heading 90][set heading 270]

        ]
      ]
    ][
      ; for cars in rescue lanes
      ifelse (random-float 1 <= prob-remove-car) [
        die ; remove a damaged car
      ][
        set damaged-inrescue-duration (damaged-inrescue-duration + 1)
        if damaged-inrescue-duration >= damaged-inrescue-duration-max [
          ; restore car to normal situation
          set speed 0.06
          if ycor <= 1 - number-of-lanes [set target-lane (3 - number-of-lanes) ]
          if ycor >= number-of-lanes - 1 [set target-lane (number-of-lanes - 3) ]
          move-to-target-lane
          set damaged-inrescue-duration 0
        ]
      ]
    ]
  ]
end

;; decrease the value of speed and patience
to slow-down-car ; car procedure
  if speed > deceleration [
    set speed speed - deceleration
    if speed < speed-min [ set speed speed-min ]
    set patience patience - 1 ; every time you hit the brakes, you loose a little patience
    if patience < 0 [set patience 0]
  ]
end

;; increase the value of speed and patience
to speed-up-car ; car procedure
  set speed speed + acceleration
  if speed > speed-top [ set speed speed-top ]
  ;set patience patience + 1
  ;if patience > max-patience [set patience max-patience]
end

;; convert a state to an integer value
to-report convert-state-int [ aState ]
  ; state-size = max-patience * 101 * world-width * 101
  report (item 0 aState) * 101 * world-width * 101 + (item 1 aState) * world-width * 101 + (item 2 aState) * 101 + (item 3 aState)
end

to create-or-remove-cars
  ; make sure we don't have too many cars for the room we have on the road
  let car-road-patches patches with [ member? pycor car-lanes ]
  if number-of-cars > count car-road-patches [
    set number-of-cars count car-road-patches
  ]

  let car-speed-seed 0.75

  create-cars (number-of-cars - count cars) [
    set size 1.0 ;0.9
    set color get-car-default-color self
    move-to one-of free-car car-road-patches
    set target-lane pycor
    set shape "car-top"
    ifelse (member? pycor car-lanes-left) [
      set heading 90
    ][
      set heading 270
    ]
    set speed car-speed-seed + random-float (speed-max - car-speed-seed)
    set speed-top  speed-max ;(2 * speed + random-float (speed-max - 2 * speed)) ;(speed-max / 2) + random-float (speed-max / 2)
    set patience (max-patience / 2) + random (max-patience / 2)
    set damaged-inlane-duration 0
    set damaged-inrescue-duration 0
    set action -1
    set reward 0
  ]

  if count cars > number-of-cars [
    let n count cars - number-of-cars
    ask n-of n [ other cars ] of selected-car [ die ]
  ]
end

to-report free-car [ car-road-patches ] ; car procedure
  let this-car self
  report car-road-patches with [
    not any? cars-here with [ self != this-car ]
  ]
end

to draw-road
  ask patches [
    set pcolor brown - random-float 0.5 ; the road is surrounded by brown ground of varying shades
  ]
  set lanes n-values number-of-lanes [ n -> number-of-lanes - (n * 2) - 1 ]

  set car-lanes n-values (number-of-lanes - 2) [ n -> number-of-lanes - (n * 2) - 3]
  set car-lanes remove-item number-of-lanes-way car-lanes
  set car-lanes-right n-values number-of-lanes-way [ n -> 2 * (number-of-lanes-way - n)]
  set car-lanes-left n-values number-of-lanes-way [ n -> 2 * (n - number-of-lanes-way)]
  ;set rescue-lanes list (number-of-lanes - 1) (1 - number-of-lanes)
  ask patches with [ abs pycor <= number-of-lanes  ] [ ; member? pycor car-lanes
    set pcolor grey - 2.5 + random-float 0.25 ; the road itself is varying shades of grey
  ]
  ask patches with [abs pycor = (number-of-lanes - 1) ] [
    set pcolor white - 4 ; rescue lane
  ]
  ask patches with [pycor = 0 ] [
    set pcolor green - 2.5 + random-float 0.25 ; median strip
  ]
  ;print (word "rescue-lanes: " rescue-lanes)
  ;print (word "car-lanes: " car-lanes)
  ;print (word "car-lanes-right: " car-lanes-right)
  ;print (word "car-lanes-left: " car-lanes-left)

  draw-road-lines
end

to draw-road-lines
  let y (last lanes) - 1 ; start below the "lowest" lane
  while [ y <= first lanes + 1 ] [
    if not member? y lanes [
      ; draw lines on road patches that are not part of a lane
      ifelse abs y = number-of-lanes [
        draw-line y yellow 0 ; yellow for the sides of the road
      ][
        ifelse ( (abs y = (number-of-lanes - 2)) or (abs y = 1) ) [
          draw-line y white 0 ; for rescue lane
        ][
          draw-line y white 0.5 ; dashed white between lanes
        ]
      ]
    ]
    set y y + 1 ; move up one patch
  ]
end

;; We use a temporary car to draw the line:
;; - with a gap of zero, we get a continuous line;
;; - with a gap greater than zero, we get a dasshed line.
to draw-line [ y line-color gap ]
  create-cars 1 [
    setxy (min-pxcor - 0.5) y
    hide-turtle
    set color line-color
    set heading 90
    repeat world-width [
      pen-up
      forward gap
      pen-down
      forward (1 - gap)
    ]
    die
  ]
end

;; choose a new lane among those with the minimum distance to your current lane (i.e., your ycor).
to choose-new-lane ; car procedure
  let other-lanes []
  if (pycor < 0) [ set other-lanes remove ycor car-lanes-left ]
  if (pycor > 0) [ set other-lanes remove ycor car-lanes-right ]
  ;let other-lanes remove ycor car-lanes

  if not empty? other-lanes [
    let min-dist min map [ y -> abs (y - ycor) ] other-lanes
    let closest-lanes filter [ y -> abs (y - ycor) = min-dist ] other-lanes
    set target-lane one-of closest-lanes
    set patience max-patience
  ]
  ;if target-lane = 0 [print ( word "choose-new-lane: target-lane = 0 !" )]
end

to move-to-target-lane ; car procedure
  ifelse (pycor < 0) [
    set heading ifelse-value target-lane < ycor [ 135 ] [ 45 ]
  ][
    if (pycor > 0) [ set heading ifelse-value target-lane < ycor [ 225 ] [ 315 ] ]
  ]

  ;set observation-distance get-safe-distance
  let blocking-cars other cars in-cone (observation-distance + abs (ycor - target-lane)) observation-angle with [ get-y-distance <= observation-distance ]
  let blocking-car-nearest min-one-of blocking-cars [ distance myself ]
  ifelse blocking-car-nearest = nobody [
    forward change-lane-step
    if get-distance-to-car get-car-ahead <= collision-distance [
      ;set color orange + 2
      print (word "collision: " self get-car-ahead)
      set nb-collisions nb-collisions + 1
    ]
    if (precision ycor 1 != 0) [set ycor precision ycor 1] ; to avoid floating point errors
  ][
    ; slow down if the car blocking us is behind, otherwise speed up
    ifelse towards blocking-car-nearest < 180 [ slow-down-car ] [ speed-up-car ]
  ]
  set color get-car-default-color self
  if (pycor < 0) [ set heading 90 ]
  if (pycor > 0) [ set heading 270 ]

  ;if ycor = 0 [print ( word "move-car-to-target-lane: current lane = 0 !" )]
end

;; allow the user to select a different car by clicking on it with the mouse
to select-car
  if mouse-down? [
    let mx mouse-xcor
    let my mouse-ycor
    if any? cars-on patch mx my [
      set selected-car one-of cars-on patch mx my
      ask selected-car [ set color red ]
      display
    ]
  ]
end

to-report get-x-distance
  report distancexy [ xcor ] of myself ycor
end

to-report get-y-distance
  report distancexy xcor [ ycor ] of myself
end

;; calculate the current exploration rate or epsilon
to-report get-epsilon ; car procedure
  ifelse driving-policy != "Greedy" [
    report exp-rate / (1 + exp-decay * ticks)
  ][
    report -1
  ]
end

; calculate the safe distance of a car, depending to its speed
; if speed = 60km/h then safe distance is 35m
; if 80km >= speed > 60km/h then safe distance is 55m
; if 100km >= speed > 80km/h then safe distance is 70m
; if 120km >= speed > 100km/h then safe distance is 100m
to-report get-safe-distance ; car procedure
  ifelse speed <= 0.6 [
    report 35 / patch-size
  ][
    ifelse speed <= 0.8[
      report 55 / patch-size
    ][
      ifelse speed <= 1.0[
        report 70 / patch-size
      ][
        report 100 / patch-size
      ]
    ]
  ]
end

;; give all cars a blueish color, but still make them distinguishable
to-report get-car-default-color [aCar]
  if aCar = selected-car [ report pink ]
  report white - random-float 5.0  ;one-of [ blue cyan sky ] + 1.5 + random-float 1.0
end

to-report get-distance-to-car [aCar]
  ifelse aCar != nobody [
    report distance aCar
  ][
    report observation-max
  ]
end

to-report get-speed-to-car [aCar]
  ifelse aCar != nobody
  [
    report [speed] of aCar
  ][
    report speed-max
  ]
end

;; report the car ahead that blocks this car in the same lane or in the neighbor lanes
to-report get-blocking-car
  let i 1
  while [ i < observation-max ]
  [
    let blocking-cars other cars in-cone i 180 with [ get-y-distance <= 1 ]
    let blocking-car min-one-of blocking-cars [ distance myself ]
    ifelse blocking-car != nobody [
      report blocking-car
    ][
      set i (i + 1)
    ]
  ]
  report nobody
end

;; report the car ahead that blocks this car in the same lane
to-report get-ahead
  let i 1
  while [ i < observation-max ]
  [
    ifelse (not any? cars-on patch-ahead i) [ ; patch-at-heading-and-distance -90 i
      set i (i + 1)
    ][
      report min-one-of cars-on patch-ahead i [ distance myself ]
    ]
  ]
  report nobody
end

;; another version of getting car ahead
to-report get-car-ahead
  let i 0
  while [ i <= observation-max ]
  [
    let blocking-cars other cars in-cone i 30 with [ get-y-distance <= 1 ]
    let blocking-car min-one-of blocking-cars [ distance myself ]
    ifelse blocking-car != nobody [
      report blocking-car
    ][
      set i (i + 1)
    ]
  ]
  report nobody
end

;; report the car behind that blocks this car in the same lane
to-report get-car-behind
  ;report min-one-of cars-here with [ xcor < [ xcor ] of myself ] [ distance myself ]
  ;ifelse here != nobody [
  ;  report here
  ;][
  let i 1
  while [ i < observation-max ]
  [
    ifelse (not any? cars-on patch-ahead (0 - i)) [ ; patch-at-heading-and-distance -90 i
      set i (i + 1)
    ][
      report min-one-of cars-on patch-ahead (0 - i) [ distance myself ]
    ]
  ]
  report nobody
  ;]

end
@#$#@#$#@
GRAPHICS-WINDOW
250
55
1478
684
-1
-1
20.0
1
10
1
1
1
0
1
0
1
-30
30
-15
15
1
1
1
ticks
30.0

BUTTON
470
15
535
50
NIL
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
540
15
605
50
go
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

BUTTON
610
15
675
50
go once
go
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

BUTTON
815
15
925
48
select car
select-car
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

MONITOR
250
690
360
735
average speed
(mean [speed] of cars with [speed > 0]) * 100
2
1
11

SLIDER
5
110
125
143
number-of-cars
number-of-cars
1
nb-cars-max
114.0
1
1
NIL
HORIZONTAL

PLOT
250
740
645
950
Speed of Cars
Time
Speed
0.0
300.0
0.0
0.5
true
true
"" ""
PENS
"average" 1.0 0 -13840069 true "" "plot mean [ speed ] of cars with [speed > 0]"
"max" 1.0 0 -955883 true "" "plot max [ speed ] of cars with [speed > 0]"
"min" 1.0 0 -1184463 true "" "plot min [ speed ] of cars with [speed > 0]"

SLIDER
5
150
125
183
acceleration
acceleration
0.001
0.01
0.005
0.001
1
NIL
HORIZONTAL

SLIDER
130
150
245
183
deceleration
deceleration
0.001
0.1
0.02
0.01
1
NIL
HORIZONTAL

PLOT
1485
105
1890
310
Driver's Patience for Cars
Time
Patience
0.0
10.0
0.0
10.0
true
true
"set-plot-y-range 0 max-patience" ""
PENS
"average" 1.0 0 -14439633 true "" "plot mean [ patience ] of cars"
"max" 1.0 0 -955883 true "" "plot max [ patience ] of cars"
"min" 1.0 0 -1184463 true "" "plot min [ patience ] of cars"

BUTTON
930
15
1040
48
follow selected car
follow selected-car
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

BUTTON
1045
15
1155
48
watch selected car
watch selected-car
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

BUTTON
1160
15
1270
48
reset perspective
reset-perspective
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

PLOT
1485
320
1890
525
Vehicles Per Lane
Time
Cars
0.0
0.0
0.0
0.0
true
true
"set-plot-y-range (floor (count turtles * 0.4)) (ceiling (count turtles * 0.6))\nforeach range length car-lanes [ i ->\n  create-temporary-plot-pen (word \"Lane \" (i + 1))\n  set-plot-pen-color item i base-colors\n]" "foreach range length car-lanes [ i ->\n  set-current-plot-pen (word \"Lane \" (i + 1))\n  plot count turtles with [ round ycor = item i car-lanes ]\n]"
PENS

SLIDER
5
190
125
223
max-patience
max-patience
10
100
50.0
1
1
NIL
HORIZONTAL

CHOOSER
5
55
245
100
driving-policy
driving-policy
"Greedy" "Greedy-CL" "-----------------------------------------------" "Nagel-Schreckenberg" "Intelligent-Driver-Model" "-----------------------------------------------" "Q-Learning" "SARSA" "Double Q-Learning" "Soft Q-Learning" "Implicit Q-Learning" "-----------------------------------------------" "Deep Q-Learning" "Deep Q-Network" "Double Deep Q-Learning" "Double Deep Q-Network" "Duel Deep Q-Network" "Prioritized ER DQN" "Deep Recurrent Q-Network" "-----------------------------------------------" "Reinforce" "Naive Actor-Critic" "Advantage Actor-Critic" "Asynchronous Advantage Actor-Critic" "Soft Actor-Critic" "-----------------------------------------------" "Proximal Policy Optimization" "Trust Region Policy Optimization" "Deep Deterministic Policy Gradient" "Twin Delayed Deep Deterministic Policy Gradient"
0

SLIDER
130
110
245
143
number-of-lanes-way
number-of-lanes-way
1
5
5.0
1
1
NIL
HORIZONTAL

SLIDER
5
230
125
263
exp-rate
exp-rate
0
1
1.0
0.01
1
NIL
HORIZONTAL

SLIDER
130
230
245
263
exp-decay
exp-decay
0
0.1
0.001
0.001
1
NIL
HORIZONTAL

SLIDER
5
270
125
303
learning-rate
learning-rate
0
0.01
0.0034
0.0001
1
NIL
HORIZONTAL

SLIDER
130
270
245
303
discount-factor
discount-factor
0.80
0.99
0.98
0.01
1
NIL
HORIZONTAL

MONITOR
1365
690
1475
735
exploration rate
get-epsilon
2
1
11

PLOT
665
740
1075
950
Selected Actions for Cars
Time
# Cars
0.0
800.0
0.0
100.0
true
true
"set-plot-y-range 0 number-of-cars" ""
PENS
"accelerate" 1.0 2 -10899396 true "" "plot count cars with [ action = 0 ]"
"decelerate" 1.0 2 -2674135 true "" "plot count cars with [ action = 2 ]"
"stay same" 1.0 2 -1184463 true "" "plot count cars with [ action = 1 ]"
"change lane" 1.0 2 -13345367 true "" "plot count cars with [ action = 3 ]"

PLOT
1080
740
1475
950
Total Reward of Cars
Time
Reward
0.0
300.0
0.0
1.0
true
false
"" ""
PENS
"reward" 1.0 0 -2674135 true "" "plot sum [reward] of cars"

MONITOR
1080
690
1185
735
total reward
sum [reward] of cars
2
1
11

MONITOR
1485
55
1590
100
average patience
mean [ patience ] of cars
2
1
11

MONITOR
665
690
745
735
% accelerate
(count cars with [ action = 0 ]) / number-of-cars * 100
2
1
11

MONITOR
770
690
850
735
% decelerate
(count cars with [ action = 2 ]) / number-of-cars * 100
2
1
11

MONITOR
885
690
965
735
% stay same
(count cars with [ action = 1 ]) / number-of-cars * 100
2
1
11

MONITOR
990
690
1075
735
% change lane
(count cars with [ action = 3 ]) / number-of-cars * 100
2
1
11

MONITOR
530
690
645
735
damaged cars
count cars with [speed = 0]
2
1
11

SLIDER
5
325
125
358
batch-size
batch-size
0
1024
128.0
32
1
NIL
HORIZONTAL

SLIDER
5
405
180
438
memory-size
memory-size
100000
10000000
100000.0
1000
1
NIL
HORIZONTAL

SWITCH
5
365
125
398
input-exp?
input-exp?
0
1
-1000

SWITCH
130
365
245
398
input-time?
input-time?
0
1
-1000

SLIDER
130
190
245
223
max-damaged-cars
max-damaged-cars
0
5
4.0
1
1
NIL
HORIZONTAL

SLIDER
5
600
175
633
simulation-time
simulation-time
10000
100000
100000.0
1000
1
NIL
HORIZONTAL

TEXTBOX
5
450
155
468
Parameters for A2C
14
0.0
1

SWITCH
5
470
135
503
multiple-workers?
multiple-workers?
1
1
-1000

TEXTBOX
5
520
155
538
Parameters for PPO
14
0.0
1

SLIDER
5
545
135
578
clip_param
clip_param
0.1
0.3
0.2
0.01
1
NIL
HORIZONTAL

SWITCH
5
645
175
678
save-experience?
save-experience?
1
1
-1000

SWITCH
5
685
175
718
dynamic-situation?
dynamic-situation?
0
1
-1000

MONITOR
1485
540
1557
585
# collisions
nb-collisions
17
1
11

PLOT
1485
590
1885
770
Number of Collisions
Time
# collisions
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"collisions" 1.0 0 -2674135 true "" "plot nb-collisions"

@#$#@#$#@
## WHAT IS IT?

This model is a more sophisticated two-lane version of the "Traffic Basic" model.  Much like the simpler model, this model demonstrates how traffic jams can form. In the two-lane version, drivers have a new option; they can react by changing lanes, although this often does little to solve their problem.

As in the Traffic Basic model, traffic may slow down and jam without any centralized cause.

## HOW TO USE IT

Click on the SETUP button to set up the cars. Click on GO to start the cars moving. The GO ONCE button drives the cars for just one tick of the clock.

The NUMBER-OF-CARS slider controls the number of cars on the road. If you change the value of this slider while the model is running, cars will be added or removed "on the fly", so you can see the impact on traffic right away.

The SPEED-UP slider controls the rate at which cars accelerate when there are no cars ahead.

The SLOW-DOWN slider controls the rate at which cars decelerate when there is a car close ahead.

The MAX-PATIENCE slider controls how many times a car can slow down before a driver loses their patience and tries to change lanes.

You may wish to slow down the model with the speed slider to watch the behavior of certain cars more closely.

The SELECT CAR button allows you to highlight a particular car. It turns that car red, so that it is easier to keep track of it. SELECT CAR is easier to use while GO is turned off. If the user does not select a car manually, a car is chosen at random to be the "selected car".

You can either [`watch`](http://ccl.northwestern.edu/netlogo/docs/dictionary.html#watch) or [`follow`](http://ccl.northwestern.edu/netlogo/docs/dictionary.html#follow) the selected car using the WATCH SELECTED CAR and FOLLOW SELECTED CAR buttons. The RESET PERSPECTIVE button brings the view back to its normal state.

The SELECTED CAR SPEED monitor displays the speed of the selected car. The MEAN-SPEED monitor displays the average speed of all the cars.

The YCOR OF CARS plot shows a histogram of how many cars are in each lane, as determined by their y-coordinate. The histogram also displays the amount of cars that are in between lanes while they are trying to change lanes.

The CAR SPEEDS plot displays four quantities over time:

- the maximum speed of any car - CYAN
- the minimum speed of any car - BLUE
- the average speed of all cars - GREEN
- the speed of the selected car - RED

The DRIVER PATIENCE plot shows four quantities for the current patience of drivers: the max, the min, the average and the current patience of the driver of the selected car.

## THINGS TO NOTICE

Traffic jams can start from small "seeds." Cars start with random positions. If some cars are clustered together, they will move slowly, causing cars behind them to slow down, and a traffic jam forms.

Even though all of the cars are moving forward, the traffic jams tend to move backwards. This behavior is common in wave phenomena: the behavior of the group is often very different from the behavior of the individuals that make up the group.

Just as each car has a current speed, each driver has a current patience. Each time the driver has to hit the brakes to avoid hitting the car in front of them, they loose a little patience. When a driver's patience expires, the driver tries to change lane. The driver's patience gets reset to the maximum patience.

When the number of cars in the model is high, drivers lose their patience quickly and start weaving in and out of lanes. This phenomenon is called "snaking" and is common in congested highways. And if the number of cars is high enough, almost every car ends up trying to change lanes and the traffic slows to a crawl, making the situation even worse, with cars getting momentarily stuck between lanes because they are unable to change. Does that look like a real life situation to you?

Watch the MEAN-SPEED monitor, which computes the average speed of the cars. What happens to the speed over time? What is the relation between the speed of the cars and the presence (or absence) of traffic jams?

Look at the two plots. Can you detect discernible patterns in the plots?

The grass patches on each side of the road are all a slightly different shade of green. The road patches, to a lesser extent, are different shades of grey. This is not just about making the model look nice: it also helps create an impression of movement when using the FOLLOW SELECTED CAR button.

## THINGS TO TRY

What could you change to minimize the chances of traffic jams forming, besides just the number of cars? What is the relationship between number of cars, number of lanes, and (in this case) the length of each lane?

Explore changes to the sliders SLOW-DOWN and SPEED-UP. How do these affect the flow of traffic? Can you set them so as to create maximal snaking?

Change the code so that all cars always start on the same lane. Does the proportion of cars on each lane eventually balance out? How long does it take?

Try using the `"default"` turtle shape instead of the car shape, either by changing the code or by typing `ask turtles [ set shape "default" ]` in the command center after clicking SETUP. This will allow you to quickly spot the cars trying to change lanes. What happens to them when there is a lot of traffic?

## EXTENDING THE MODEL

The way this model is written makes it easy to add more lanes. Look for the `number-of-lanes` reporter in the code and play around with it.

Try to create a "Traffic Crossroads" (where two sets of cars might meet at a traffic light), or "Traffic Bottleneck" model (where two lanes might merge to form one lane).

Note that the cars never crash into each other: a car will never enter a patch or pass through a patch containing another car. Remove this feature, and have the turtles that collide die upon collision. What will happen to such a model over time?

## NETLOGO FEATURES

Note the use of `mouse-down?` and `mouse-xcor`/`mouse-ycor` to enable selecting a car for special attention.

Each turtle has a shape, unlike in some other models. NetLogo uses `set shape` to alter the shapes of turtles. You can, using the shapes editor in the Tools menu, create your own turtle shapes or modify existing ones. Then you can modify the code to use your own shapes.

## RELATED MODELS

- "Traffic Basic": a simple model of the movement of cars on a highway.

- "Traffic Basic Utility": a version of "Traffic Basic" including a utility function for the cars.

- "Traffic Basic Adaptive": a version of "Traffic Basic" where cars adapt their acceleration to try and maintain a smooth flow of traffic.

- "Traffic Basic Adaptive Individuals": a version of "Traffic Basic Adaptive" where each car adapts individually, instead of all cars adapting in unison.

- "Traffic Intersection": a model of cars traveling through a single intersection.

- "Traffic Grid": a model of traffic moving in a city grid, with stoplights at the intersections.

- "Traffic Grid Goal": a version of "Traffic Grid" where the cars have goals, namely to drive to and from work.

- "Gridlock HubNet": a version of "Traffic Grid" where students control traffic lights in real-time.

- "Gridlock Alternate HubNet": a version of "Gridlock HubNet" where students can enter NetLogo code to plot custom metrics.

## HOW TO CITE

If you mention this model or the NetLogo software in a publication, we ask that you include the citations below.

For the model itself:

* Wilensky, U. & Payette, N. (1998).  NetLogo Traffic 2 Lanes model.  http://ccl.northwestern.edu/netlogo/models/Traffic2Lanes.  Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.

Please cite the NetLogo software as:

* Wilensky, U. (1999). NetLogo. http://ccl.northwestern.edu/netlogo/. Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.

## COPYRIGHT AND LICENSE

Copyright 1998 Uri Wilensky.

![CC BY-NC-SA 3.0](http://ccl.northwestern.edu/images/creativecommons/byncsa.png)

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.  To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/3.0/ or send a letter to Creative Commons, 559 Nathan Abbott Way, Stanford, California 94305, USA.

Commercial licenses are also available. To inquire about commercial licenses, please contact Uri Wilensky at uri@northwestern.edu.

This model was created as part of the project: CONNECTED MATHEMATICS: MAKING SENSE OF COMPLEX PHENOMENA THROUGH BUILDING OBJECT-BASED PARALLEL MODELS (OBPML).  The project gratefully acknowledges the support of the National Science Foundation (Applications of Advanced Technologies Program) -- grant numbers RED #9552950 and REC #9632612.

This model was converted to NetLogo as part of the projects: PARTICIPATORY SIMULATIONS: NETWORK-BASED DESIGN FOR SYSTEMS LEARNING IN CLASSROOMS and/or INTEGRATED SIMULATION AND MODELING ENVIRONMENT. The project gratefully acknowledges the support of the National Science Foundation (REPP & ROLE programs) -- grant numbers REC #9814682 and REC-0126227. Converted from StarLogoT to NetLogo, 2001.

<!-- 1998 2001 Cite: Wilensky, U. & Payette, N. -->
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car-left
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

car-right
false
0
Polygon -7500403 true true 0 180 21 164 39 144 60 135 74 132 87 106 97 84 115 63 141 50 165 50 225 60 300 150 300 165 300 225 0 225 0 180
Circle -16777216 true false 30 180 90
Circle -16777216 true false 180 180 90
Polygon -16777216 true false 138 80 168 78 166 135 91 135 106 105 111 96 120 89
Circle -7500403 true true 195 195 58
Circle -7500403 true true 47 195 58

car-top
true
0
Polygon -7500403 true true 151 8 119 10 98 25 86 48 82 225 90 270 105 289 150 294 195 291 210 270 219 225 214 47 201 24 181 11
Polygon -16777216 true false 210 195 195 210 195 135 210 105
Polygon -16777216 true false 105 255 120 270 180 270 195 255 195 225 105 225
Polygon -16777216 true false 90 195 105 210 105 135 90 105
Polygon -1 true false 205 29 180 30 181 11
Line -7500403 true 210 165 195 165
Line -7500403 true 90 165 105 165
Polygon -16777216 true false 121 135 180 134 204 97 182 89 153 85 120 89 98 97
Line -16777216 false 210 90 195 30
Line -16777216 false 90 90 105 30
Polygon -1 true false 95 29 120 30 119 11

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.4.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
1
@#$#@#$#@
