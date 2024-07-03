; The width of a lane: 3.5m (or 3.75m) ~ 2 * (patch-size)
; The width of a car: ~1.7m

extensions [ py ]

breed [cars car]

globals [
  selected-car          ; the currently selected car
  nb-cars-max           ; maximum number of car
  id-count              ; temp value to set id for cars

  nb-lanes              ; number of lanes for two-way
  lanes                 ; a list of the y coordinates of different lanes
  car-lanes
  car-lanes-left        ; a list of the y coordinates of different lanes for cars (from left to right)
  car-lanes-right       ; a list of the y coordinates of different lanes for cars (from right to left)
  ;rescue-lanes         ; a list of the y coordinates of rescue lanes

  speed-max             ; max speed in highway
  speed-min             ; min speed in highway
  speed-ratio           ; the ratio to multiply with speed

  hard-brake-speed      ; the speed that a car has when it must to have a hard-brake
  hard-brake-distance   ; the distance that speed must to have a hard-brake

  collision-distance    ; the distance that can be considered as collission

  damaged-color                 ; the color for damaged car
  damaged-max-duration-inlane   ; the maximum duration for a damaged car stand on running lanes
  damaged-max-duration-inrescue ; the maximum duration for a damaged car stand on rescue lanes
  damaged-nb-cars-inlane        ; the maximum number of damaged cars who stand on running lanes

  speed-change-lane      ; the value that we forward a car when it changes lane
  move-rescue-lane-step  ; the value that we forward a damaged car to the rescue lanes

  prob-damaged-car      ; the rate at which a car is damaged when traveling on a road
  prob-add-car          ; the probability that we add a car
  prob-remove-car       ; the probability that we remove a damaged car

  observation-max       ; the maximum distance that a car can observe
  observation-blocking  ; the distance that we use to observe blocking cars
  observation-angle     ; the angle that we use to observe blocking cars

  nb-collisions         ; number of collisions

  ; parameters for rl algorithms
  state-env
  ac-update-steps
]

; define attributes for car agent
cars-own [
  speed                       ; the current speed of the car
  speed-top                   ; the maximum speed of the car (different for all cars)
  patience                    ; the driver's max number of patience
  patience-top                ; the driver's max number of patience
  target-lane                 ; the desired lane of the car
  damaged-duration-inlane     ; the duration that a damaged car is in running lanes
  damaged-duration-inrescue   ; the duration that a damaged car is in rescue lanes

  init-xcor                   ; the initial x coordinate for REINFORCE algo
  id

  ; attributes for rl
  reward
  state
  action                      ; 0 = accelerate, 1 = stay same, 2 = decelerate, 3 = change lane
  next-state
]

to setup
  clear-all

  set nb-lanes (nb-lanes-oneway * 2 + 3)
  set nb-cars-max round (nb-lanes-oneway * 2 * world-width * 0.25)
  set id-count 0

  set speed-max 1.0 ; ~ 120 km/h
  set speed-min 0.0 ;
  set speed-ratio 0.35 ; need to recalculate

  set hard-brake-speed 0.02
  set hard-brake-distance 1.5

  set collision-distance  1

  set damaged-color red
  set damaged-max-duration-inlane  1800  ; 3600 ~ 30 minutes
  set damaged-max-duration-inrescue 1800 ; 3600 ~ 45 minutes
  set damaged-nb-cars-inlane 0

  set speed-change-lane 0.08
  set move-rescue-lane-step speed-change-lane * 0.1 ;0.06

  set prob-damaged-car 0.00002
  set prob-add-car 0.002
  set prob-remove-car 0.0006

  set observation-max world-width / 3
  set observation-blocking 2.75
  set observation-angle 45

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

    py:set "action_size" 4   ; 0 = accelerate, 1 = stay same, 2 = decelerate, 3 = change lane
    py:set "gamma" discount-factor
    py:set "alpha" learning-rate

    if (driving-policy = "SARSA") or
       (driving-policy = "Q-Learning") or
       (driving-policy = "Double Q-Learning") [
      setup-tabular-algos
    ]

    if (driving-policy = "Deep Q-Learning") or
       (driving-policy = "Deep Q-Network") or
       (driving-policy = "Double Deep Q-Network") [
      setup-approximate-algos
    ]

    if (driving-policy = "Naive Actor-Critic") or
       (driving-policy = "Advantage Actor-Critic") or
       (driving-policy = "Soft Actor-Critic") [
      setup-actor-critic-algos
    ]

    if (driving-policy = "Proximal Policy Optimization")  [
      setup-ppo
    ]

    if (driving-policy = "Reinforce")  [
      setup-reinforce
    ]
  ]

  reset-ticks
end

to go
  if ticks >= simulation-time [stop]

  ; add cars
  if dynamic-situation? [
    if (random-float 1 <= prob-add-car) and (nb-cars < nb-cars-max) [
      set nb-cars (nb-cars + 1)
      ;print "Add 1 car"
    ]
  ]

  create-or-remove-cars

  if ( (driving-policy = "Greedy") or (driving-policy = "Greedy-CL") ) [
    ;ask cars with [speed > 0] [ move-forward-gd ]
    ;set nb-collisions 0
    ask cars [
      move-forward-gd
      ;if get-distance get-ahead < 1 [
      ;  set nb-collisions (nb-collisions + 1)
      ;]
    ]
  ]

  if (driving-policy = "SARSA") or
     (driving-policy = "Q-Learning") [
    go-sarsa-ql
  ]

  if (driving-policy = "Double Q-Learning") [
    go-double-ql
  ]

  if (driving-policy = "Deep Q-Learning") or
     (driving-policy = "Deep Q-Network") or
     (driving-policy = "Double Deep Q-Network") [
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

  if (driving-policy = "Reinforce") [
    go-reinforce
  ]

  tick
end

;; setup function for tabular RL algorithms (i.e., Q-Learning, Double Q-Learning, SARSA)
to setup-tabular-algos
  set state-env (list
    [-> patience]
    [-> round (100 * speed)]
    [-> round (get-distance get-ahead)]
    [-> round (100 * get-speed get-ahead)])

  let state-size  (patience-max + 1) * 101 * world-width * 101; speed: 0 -> 100 -- distance: 0 - world-width -- speed-car-ahead: 0 -> 100
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
    [-> get-distance get-ahead]
    [-> get-speed get-ahead]
    [-> get-distance get-behind]
    [-> get-speed get-behind])

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

  if (driving-policy = "Deep Q-Network") or
     (driving-policy = "Double Deep Q-Network") [
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
    [-> get-distance get-ahead]
    [-> get-speed get-ahead]
    [-> get-distance get-behind]
    [-> get-speed get-behind])

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
    py:set "update_offset" (round (ac-update-steps / nb-cars) )
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
    [-> get-distance get-ahead]
    [-> get-speed get-ahead]
    [-> get-distance get-behind]
    [-> get-speed get-behind])

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
    py:set "update_offset" (round (ac-update-steps / nb-cars) )
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

;;setup reinforce algorithm: Monte-Carlo Policy-Gradient Control
to setup-reinforce
  set state-env (list
    [-> speed]
    [-> get-distance get-ahead]
    [-> get-speed get-ahead]
    [-> get-distance get-behind]
    [-> get-speed get-behind])

  if input-exp? [
    set state-env lput [-> get-epsilon ] state-env
  ]

  if input-time? [
    set state-env lput [-> ticks] state-env
  ]

  py:set "input_size" length state-env
  py:set "hl_size" 36
  (py:run
    "import tensorflow as tf"
    "import numpy as np"
    "import tensorflow_probability as tfp"
    "from tensorflow import keras"
    "from keras import layers"
    "R_network = keras.Sequential()"
    "R_network.add(layers.Dense(hl_size, input_shape=(input_size,), activation='relu'))"
    "R_network.add(layers.Dense(hl_size, activation='relu'))"
    "R_network.add(layers.Dense(hl_size, activation='relu'))"
    "R_network.add(layers.Dense(action_size, activation='softmax'))"
    "optimizer = keras.optimizers.Adam(learning_rate=alpha)")

  py:set "nb_cars" ((max [ who ] of turtles) + 1)
  ; 4 matrices to store state, action, reward
  (py:run
    "mem_states = [[] for i in range(nb_cars)]"
    "mem_rewards = [[] for i in range(nb_cars)]"
    "mem_actions = [[] for i in range(nb_cars)]"
    "mem_returns = [[] for i in range(nb_cars)]")
end

;; go function for Double Q-Learning algorithm
;; 18/01/2023: modify this to use only one agent to update Q table ???
to go-sarsa-ql
  ask cars [
    ifelse speed > 0 [
      set state map runresult state-env ; get current state
      let state-int convert-state-int state
      py:set "state" state-int

      ; choose an action corresponding current state, using e-greedy
      ifelse (0.05 + random-float 1) < get-epsilon [
        ifelse (greedy-attitude?)[
          set action 0 ; greedy attitude -> accelerate
        ][
          set action random 4 ; explore the random / new action
          while [action = 3 and patience > 0] [set action random 4]
        ]
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
    ][
      ; for car with speed = 0
      handle-damaged-car
    ]
  ]
end

;; go function for Double Q-Learning algorithm
;; 18/01/2023: modify this to use only one agent to update Q table ???
to go-double-ql
  ask cars [
    ifelse speed > 0 [
      set state map runresult state-env ; get current state
      let state-int convert-state-int state
      py:set "state" state-int

      ; choose an action corresponding current state, using e-greedy
      ifelse (0.05 + random-float 1) < get-epsilon [
        ifelse (greedy-attitude?)[
          set action 0 ; greedy attitude -> accelerate
        ][
          set action random 4 ; explore the random / new action
          while [action = 3 and patience > 0] [set action random 4]
        ]
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
        py:run "QA[state,action] += alpha * (reward + gamma * QB[next_state,next_action] - QA[state,action])"
      ][ ; update QB
        py:run "next_action = np.argmax(QB[next_state,:])"
        py:run "QB[state,action] += alpha * (reward + gamma * QA[next_state,next_action] - QB[state,action])"
      ]
    ][
      ; for car with speed = 0
      handle-damaged-car
    ]
  ]
end

;; go for Deep Q-Learning, Deep Q-Network and Double Deep Q-Network
to go-approximate-algos
  ; get current state for all cars
  ask cars with [speed > 0] [
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
  ask cars with [speed > 0] [
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
    py:run "q_values = np.max(Q_network.predict(next_states), axis = 1)" ; axis = 1 means to find max value along rows
  ]

  if driving-policy = "Deep Q-Network" [
    py:run "q_values = np.max(Q_hat_network.predict(next_states), axis = 1)" ; axis = 1 means to find max value along rows
  ]

  if driving-policy = "Double Deep Q-Network"  [
    (py:run
      "next_targets = Q_network.predict(next_states)"
      "best_next_actions = np.argmax(next_targets, axis=1)"
      "next_targets_hat = Q_hat_network.predict(next_states)"
      "q_values = next_targets_hat[np.arange(len(next_targets_hat)), best_next_actions]"
    )
  ]
  py:run "targets[np.arange(targets.shape[0]), actions] = rewards + gamma*q_values"
  py:run "Q_network.train_on_batch(states, targets)"

  if (driving-policy = "Deep Q-Network") or
     (driving-policy = "Double Deep Q-Network") [
    (py:run
      "step = step + 1"
      "if (step % update_steps == 0): Q_hat_network = keras.models.clone_model(Q_network)")
  ]
end

;; go for Naive Actor-Critic
;; Naive AC: update neural networks on each timestamp
to go-nac
  ; get current states
  ask cars with [speed > 0] [
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
  ask cars with [speed > 0] [
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
  ask cars with [speed > 0] [
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
  ask cars with [speed > 0] [
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
  ask cars with [speed > 0] [
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
  ask cars with [speed > 0] [
    move-forward-rl
    set next-state map runresult state-env
  ]

  ; stores current state, action, reward for each car
  py:run "step = step + 1"
  ;; 18/01/2023: modify this to update 1 time by only one agent ???
  ask cars [
    py:set "id" id
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
  ask cars with [speed > 0] [
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
  ask cars with [speed > 0] [
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

;;go for REINFORCE
to go-reinforce
  ; get current states
  ask turtles [
    set state map runresult state-env
  ]
  let turtle-list sort turtles
  py:set "states" map [ t -> [ state ] of t ] turtle-list

  let actions py:runresult "np.argmax(R_network.predict(np.array(states)), axis = 1)"
  (foreach turtle-list actions [ [t a] ->
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
  ask turtles [
    move-forward-rl
    set next-state map runresult state-env
  ]

   ; stores current state, action, reward for each turtle
  ask turtles [
    py:set "id" id
    py:set "state" state
    py:set "action" action
    py:set "reward" reward

    ifelse pxcor != init-xcor [
      py:run "mem_states[id].append(state)"
      py:run "mem_actions[id].append(action)"
      py:run "mem_rewards[id].append(reward)"
    ][
      ; return the initial x coordinate
      (py:run
        "mem_returns[id] = []"
        "sum_reward = 0"
        "mem_returns[id].reverse()"
        "for r in mem_rewards[id]:"
        "   sum_reward = r + gamma*sum_reward"
        "   mem_returns[id].append(sum_reward)"
        "mem_returns[id].reverse()"
        "mem_states[id] = np.array(mem_states[id], dtype=np.float32)"
        "mem_actions[id] = np.array(mem_actions[id], dtype=np.float32)"
        "mem_returns[id] = np.array(mem_returns[id], dtype=np.float32)"
        "for a_state, a_return, a_action in zip(mem_states[id], mem_returns[id], mem_actions[id]):"
        "   with tf.GradientTape() as tape:"
        "      prob = R_network(np.array([a_state]), training=True)"
        "      dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)"
        "      log_prob = dist.log_prob(a_action)"
        "      loss = -log_prob*a_return"
        "   grads = tape.gradient(loss, R_network.trainable_variables)"
        "   r_optimizer.apply_gradients(zip(grads, R_network.trainable_variables))"
        ;; reset
        "mem_states[id] = []"
        "mem_actions[id] = []"
        "mem_rewards[id] = []"
      )
    ]
  ]

end

;; forward car according rl algorithms, with speed > 0
to move-forward-rl ; car procedure
  ifelse (prob-damaged-car <= random-float 1) [
    ;ifelse (member? pycor car-lanes-left) [ set heading 90 ][ set heading 270 ]
    if (pycor < 0) [ set heading 90 ]
    if (pycor > 0) [ set heading 270 ]

    if action = 0 [ speed-up ]   ; 0 = accelerate, green
    if action = 1 [ set color yellow ]           ; 1 = stay same, yellow
    if action = 2 [ slow-down ]     ; 2 = decelerate, red
    if action = 3 [                              ; 3 = change lane, blue
      choose-new-lane
      if ycor != target-lane [
        change-lane
      ]
    ]

    handle-running-car

    forward speed * speed-ratio

    set reward get-reward
  ][
    if (damaged-nb-cars-inlane < max-damaged-cars) [ ; this car will stop
      set speed 0
      set color damaged-color
      set damaged-duration-inlane 1
      set damaged-nb-cars-inlane (damaged-nb-cars-inlane + 1)
    ]
  ]
end


to-report get-reward ; car procedure
  ifelse get-distance get-ahead < 1 ; a collission, added 06/05/2024
  [
    report -100  ; if having a collision: -100
  ][
    ; if having a hard-brake: -10
  ]

  report speed + (patience-top - patience) / patience-top

  ;report (log (speed + 1e-8) 2) + (log ((patience - patience-current) / patience + 1e-8) 2) ; reward can be positive or negative
end

;; forward car according greedy algorithm
;; need to take into account the inertia (10/05/2024)
to move-forward-gd ; car procedure
  ifelse speed > 0 [ ; for car with speed > 0
    ifelse (prob-damaged-car < random-float 1) [
      ;ifelse ycor != target-lane [
      ;  change-lane ; move to target lane
      ;][
        if (pycor < 0) [ set heading 90 ]
        if (pycor > 0) [ set heading 270 ]

        set color get-default-color self

        handle-running-car

        if patience <= 0  [ ; want to change lane
          choose-new-lane
          change-lane ; move to target lane
        ]
      ;]
    ][
      if (damaged-nb-cars-inlane < max-damaged-cars) [ ; this car will stop
        set speed 0
        set color damaged-color
        set damaged-duration-inlane 1
        set damaged-nb-cars-inlane (damaged-nb-cars-inlane + 1)
      ]
    ]
  ][ ; for car with speed = 0
    handle-damaged-car
  ]
end

; handle blocking car, in case that car is running
to handle-blocking-car-running [blocking-car]
  ;while [speed >= ([ speed ] of blocking-y-car)] [slow-down]
  ;set speed [ speed ] of blocking-car ; match the speed of the car ahead of you and then slow down so you are driving a bit slower than that car.

  ; distance <= collision-distance: collission
  ; hard-brake-distance < distance <= slow-down-distance: hard-brake
  ; slow-down-distance < distance <= safe-distace: slow-down
  ; safe-distance < distance : can speed up
  let d get-distance blocking-car
  ifelse d < collision-distance [
      ;print (word "collission: " self)
      set color orange + 2
      set nb-collisions nb-collisions + 1
  ][
    ifelse d < hard-brake-distance [
      hard-brake
    ][
      ifelse d < get-safe-distance [
        slow-down
      ][
        ifelse d < observation-max [
          set speed [speed] of blocking-car
        ][
          speed-up ; tentatively speed up
        ]
      ]
    ]
  ]
end

; handle blocking car, in case that car is damaged
to handle-blocking-car-damaged [blocking-car]
  let d get-distance blocking-car
  if d < get-safe-distance [ choose-new-lane ]

  ifelse d < hard-brake-distance [ ; must have a hard-brake
    hard-brake ; inertia
  ][
    ;while [speed > hard-brake-speed] [slow-down]
    slow-down
  ]
  ; set speed hard-brake-speed
  if (ycor != target-lane) [
     change-lane
  ]
end

; handle running car
to handle-running-car ; car procedure
  ;let blocking-y-cars other cars in-cone (observation-blocking + speed) observation-angle with [ get-x-distance <= observation-blocking ]
  ;set observation-blocking get-safe-distance
  ;let blocking-cars other cars in-cone observation-blocking observation-angle
  ;let blocking-car min-one-of blocking-cars [ distance myself ]
  ;let blocking-car min-one-of cars-on patch-ahead observation-blocking [ distance myself ]
  let blocking-car get-ahead
  ifelse blocking-car != nobody [
    ifelse [ speed ] of blocking-car > 0 [
      handle-blocking-car-running blocking-car
    ][
      handle-blocking-car-damaged blocking-car
    ]
  ][
    speed-up
  ]
end

; handle the damaged cars, with speed = 0
to handle-damaged-car ; car procedure
  if (dynamic-situation?) [
    ifelse ((ycor > 1 - nb-lanes) and (ycor < nb-lanes - 1))[
      set damaged-duration-inlane (damaged-duration-inlane + 1)
      if damaged-duration-inlane >= damaged-max-duration-inlane [ ; move car to rescue lane
        if (ycor < 0) [ set heading 180 ]
        if (ycor > 0) [ set heading 0 ]

        if not any? cars-on patch-ahead 1 [
          forward move-rescue-lane-step * speed-ratio
        ]

        if ( (ycor <= 1 - nb-lanes) or (ycor >= nb-lanes - 1) ) [
          set damaged-duration-inlane 0
          set damaged-duration-inrescue 1
          set damaged-nb-cars-inlane (damaged-nb-cars-inlane - 1)
        ]
      ]
    ][
      ; for cars in rescue lanes
      ifelse (random-float 1 <= prob-remove-car) [
        ;print "Remove a damaged car"
        die ; remove a damaged car
      ][
        set damaged-duration-inrescue (damaged-duration-inrescue + 1)
        if damaged-duration-inrescue >= damaged-max-duration-inrescue [
          ; restore car to normal situation
          set speed 0.06
          if ycor <= 1 - nb-lanes [set target-lane (3 - nb-lanes) ]
          if ycor >= nb-lanes - 1 [set target-lane (nb-lanes - 3) ]
          change-lane
          set damaged-duration-inrescue 0
        ]
      ]
    ]
  ]
end

;; slow down smoothly: decrease the value of speed and patience
to slow-down ; car procedure
  set color yellow
  if speed > deceleration [
    set speed (speed - deceleration)
    if speed < speed-min [ set speed speed-min ]

    if patience > 0 [
      set patience patience - 1 ; every time you hit the brakes, you loose a little patience
    ]
  ]

  forward speed * speed-ratio
end

;; hard brake in danger situations
to hard-brake ; car procedure
  set color orange + 1
  if speed > 0 [
    set speed hard-brake-speed
    set patience patience - 5 ; every time you hit the brakes, you loose a little patience
    if patience < 0 [set patience 0]
  ]

  forward speed * speed-ratio
end

;; speed-up smoothly: increase the value of speed and patience
to speed-up ; car procedure
  set color green
  set speed (speed + acceleration)
  if speed > speed-top [ set speed speed-top ]

  set patience patience + 1
  if patience > patience-top [
    set patience patience-top
    set target-lane ycor
  ]

  forward speed * speed-ratio
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
    set patience patience-top
  ]
  ;if target-lane = 0 [print ( word "choose-new-lane: target-lane = 0 !" )]
end

; move to the target lane, need slow down
to change-lane ; car procedure
  set color blue + 1
  ifelse (pycor < 0) [
    set heading ifelse-value target-lane < ycor [ 135 ] [ 45 ]
  ][
    if (pycor > 0) [ set heading ifelse-value target-lane < ycor [ 225 ] [ 315 ] ]
  ]

  set observation-blocking get-safe-distance
  let blocking-cars other cars in-cone (observation-blocking + abs (ycor - target-lane)) observation-angle with [ get-y-distance <= observation-blocking ]
  let blocking-car-nearest min-one-of blocking-cars [ distance myself ]
  ifelse blocking-car-nearest = nobody [
    forward speed-change-lane * speed-ratio
    if (precision ycor 1 != 0) [set ycor precision ycor 1] ; to avoid floating point errors
  ][
    ; slow down if the car blocking us is behind, otherwise speed up
    ifelse towards blocking-car-nearest < 180 [ slow-down ] [ speed-up ]
  ]
  if (ycor = target-lane) [ set color get-default-color self ]
  if (pycor < 0) [ set heading 90 ]
  if (pycor > 0) [ set heading 270 ]

  ;if ycor = 0 [print ( word "move-car-to-target-lane: current lane = 0 !" )]
  ;print (word "change lane: " self)
end


;; calculate accelartion according IDM (Intelligent Driver Model) added 28/08/2023
to-report cal-accele-idm
  report 0
end

;; calculate accelartion according OVM (Optimal Velocity Model)
to-report cal-accele-ovm
  report 0
end

;; convert a state to an integer value
to-report convert-state-int [ aState ]
  ; state-size = max-patience * 101 * world-width * 101
  report (item 0 aState) * 101 * world-width * 101 + (item 1 aState) * world-width * 101 + (item 2 aState) * 101 + (item 3 aState)
end

to create-or-remove-cars
  ; make sure we don't have too many cars for the room we have on the road
  let car-road-patches patches with [ member? pycor car-lanes ]
  if nb-cars > count car-road-patches [
    set nb-cars count car-road-patches
  ]

  let car-speed-seed 0.70

  create-cars (nb-cars - count cars) [
    set size 0.9
    set color get-default-color self
    move-to one-of free-car car-road-patches
    set target-lane pycor
    ifelse (member? pycor car-lanes-left) [
      set shape "car-left"
      set heading 90
    ][
      set shape "car-right"
      set heading 270
    ]
    set speed car-speed-seed + random-float (speed-max - car-speed-seed)
    set speed-top  speed-max ;(2 * speed + random-float (speed-max - 2 * speed)) ;(speed-max / 2) + random-float (speed-max / 2)
    set patience-top (patience-max / 2) + random (patience-max / 2)
    set patience patience-top
    set damaged-duration-inlane 0
    set damaged-duration-inrescue 0
    set action -1
    set reward 0

    set init-xcor pxcor
    set id id-count
    set id-count id-count + 1
  ]

  if count cars > nb-cars [
    let n count cars - nb-cars
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
  set lanes n-values nb-lanes [ n -> nb-lanes - (n * 2) - 1 ]

  set car-lanes n-values (nb-lanes - 2) [ n -> nb-lanes - (n * 2) - 3]
  set car-lanes remove-item nb-lanes-oneway car-lanes
  set car-lanes-right n-values nb-lanes-oneway [ n -> 2 * (nb-lanes-oneway - n)]
  set car-lanes-left n-values nb-lanes-oneway [ n -> 2 * (n - nb-lanes-oneway)]
  ;set rescue-lanes list (nb-lanes - 1) (1 - nb-lanes)
  ask patches with [ abs pycor <= nb-lanes  ] [ ; member? pycor car-lanes
    set pcolor grey - 2.5 + random-float 0.25 ; the road itself is varying shades of grey
  ]
  ask patches with [abs pycor = (nb-lanes - 1) ] [
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
      ifelse abs y = nb-lanes [
        draw-line y yellow 0 ; yellow for the sides of the road
      ][
        ifelse ( (abs y = (nb-lanes - 2)) or (abs y = 1) ) [
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
    ifelse speed <= 0.8 [
      report 55 / patch-size
    ][
      ifelse speed <= 1.0 [
        report 70 / patch-size
      ][
        report 100 / patch-size
      ]
    ]
  ]
end

;; give all cars a blueish color, but still make them distinguishable
to-report get-default-color [aCar]
  if aCar = selected-car [ report pink ]
  report white - random-float 5.0  ;one-of [ blue cyan sky ] + 1.5 + random-float 1.0
end

to-report get-x-distance
  report distancexy [ xcor ] of myself ycor
end

to-report get-y-distance
  report distancexy xcor [ ycor ] of myself
end

; get distance to a car
to-report get-distance [aCar]
  ifelse aCar != nobody [
    report distance aCar
  ][
    report observation-max
  ]
end

; get speed of a car
to-report get-speed [aCar]
  ifelse aCar != nobody
  [
    report [speed] of aCar
  ][
    report speed-max
  ]
end

;; get the car ahead that blocks this car in the same lane
to-report get-ahead
  let here nobody
  ifelse (pycor < 0) [
    set here min-one-of cars-here with [ xcor > [ xcor ] of myself ] [ distance myself ]
  ][
    ;if (pycor > 0) [
    set here min-one-of cars-here with [ xcor < [ xcor ] of myself ] [ distance myself ]
    ;]
  ]

  if here != nobody [ report here ]

  let d 1
  while [ d < observation-max ]
  [
    ifelse patch-ahead d != nobody [
      report min-one-of cars-on patch-ahead d [ distance myself ]
    ][
      set d (d + 1)
    ]
  ]

  report nobody
end

;; get the car behind that blocks this car in the same lane
to-report get-behind
  let here nobody
  ifelse (pycor < 0) [
    set here min-one-of cars-here with [ xcor < [ xcor ] of myself ] [ distance myself ]
  ][
    ;if (pycor > 0) [
    set here min-one-of cars-here with [ xcor > [ xcor ] of myself ] [ distance myself ]
    ;]
  ]

  if here != nobody [ report here ]

  let d 1
  while [ d < observation-max ]
  [
    set here min-one-of cars-on patch-ahead (0 - d) [ distance myself ]
    ifelse here != nobody [
      report here
    ][
      set d (d + 1)
    ]
  ]
  report nobody
end

;; report the car ahead that blocks this car in the same lane or in the neighbor lanes
;to-report get-blocking-car
;  let i 1
;  while [ i < observation-max ]
;  [
;    let blocking-cars other cars in-cone i 180 with [ get-y-distance <= 1 ]
;    let blocking-car min-one-of blocking-cars [ distance myself ]
;    ifelse blocking-car != nobody [
;      report blocking-car
;    ][
;      set i (i + 1)
;    ]
;  ]
;  report nobody
;end

;to-report get-nb-lanes
  ; To make the number of lanes easily adjustable, remove this
  ; reporter and create a slider on the interface with the same
  ; name. 8 lanes is the maximum that currently fit in the view.
;  report 3
;end
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
10
535
45
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
10
605
45
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
10
675
45
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
10
925
43
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
1485
530
1595
575
average speed
(mean [speed] of cars with [speed > 0]) * 100
2
1
11

SLIDER
5
70
125
103
nb-cars
nb-cars
1
nb-cars-max
96.0
1
1
NIL
HORIZONTAL

PLOT
1485
580
1875
760
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
"average" 1.0 0 -11085214 true "" "plot mean [ speed ] of cars with [speed > 0]"
"max" 1.0 0 -817084 true "" "plot max [ speed ] of cars with [speed > 0]"
"min" 1.0 0 -1184463 true "" "plot min [ speed ] of cars with [speed > 0]"

SLIDER
5
110
125
143
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
110
245
143
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
1875
280
Patience coefficient
Time
Patience
0.0
10.0
0.0
10.0
true
true
"set-plot-y-range 0 patience-max" ""
PENS
"average" 1.0 0 -13840069 true "" "plot mean [ patience ] of cars"
"max" 1.0 0 -817084 true "" "plot max [ patience ] of cars"
"min" 1.0 0 -1184463 true "" "plot min [ patience ] of cars"

BUTTON
930
10
1040
43
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
10
1155
43
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
10
1270
43
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

SLIDER
5
150
125
183
patience-max
patience-max
10
100
50.0
1
1
NIL
HORIZONTAL

CHOOSER
5
10
245
55
driving-policy
driving-policy
"Greedy" "Greedy-CL" "--------------------Car Following Models-----------------------" "Nagel-Schreckenberg" "Intelligent Driver Model (IDM)" "Adaptive Cruise Control" "Optimal Velocity Model (OVM)" "------------------------Tabular RL-----------------------------" "Q-Learning" "n-steps Q-Learning" "Conservative Q-Learning" "SARSA" "n-steps SARSA" "Double Q-Learning" "Soft Q-Learning" "Implicit Q-Learning" "-------------------------Deep RL--------------------------------" "Deep Q-Learning" "Slate Q-Learning" "Deep Q-Network" "Double Deep Q-Learning" "Double Deep Q-Network" "Deep Recurrent Q-Network" "Deep Attention Recurrent Q-Network" "Parametric Deep Q-Network" "Dueling Deep Q-Network" "Bootstrapped Deep Q-Network" "Categorial Deep Q-Network" "Ensemble Deep Q-Network" "Thompson Deep Q-Network" "QR Deep Q-Network" "Prioritized ER DQN" "---------------------------------------------------------------" "Reinforce" "Naive Actor-Critic" "Advantage Actor-Critic" "Asynchronous Advantage Actor-Critic" "Soft Actor-Critic" "---------------------------------------------------------------" "Proximal Policy Optimization (PPO)" "Asynchronous Proximal Policy Optimization (APPO)" "Decentralized Distributed Proximal Policy Optimization (DD-PPO)" "Trust Region Policy Optimization" "Deep Deterministic Policy Gradient (DDPG)" "Twin Delayed Deep Deterministic Policy Gradient"
0

SLIDER
130
70
245
103
nb-lanes-oneway
nb-lanes-oneway
1
5
5.0
1
1
NIL
HORIZONTAL

SLIDER
5
200
125
233
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
200
245
233
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
240
125
273
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
240
245
273
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
640
690
750
735
exploration rate
get-epsilon
2
1
11

PLOT
880
740
1355
920
Selected Actions for Cars
Time
# Cars
0.0
800.0
0.0
100.0
true
true
"set-plot-y-range 0 nb-cars" ""
PENS
"accelerate" 1.0 2 -10899396 true "" "plot count cars with [ action = 0 ]"
"decelerate" 1.0 2 -408670 true "" "plot count cars with [ action = 2 ]"
"stay same" 1.0 2 -9276814 true "" "plot count cars with [ action = 1 ]"
"change lane" 1.0 2 -13345367 true "" "plot count cars with [ action = 3 ]"

PLOT
345
740
750
920
Reward of Cars
Time
Reward
0.0
300.0
0.0
1.0
true
true
"" ""
PENS
"average" 1.0 0 -13840069 true "" "plot mean [reward] of cars with [speed > 0]"
"max" 1.0 0 -817084 true "" "plot max [reward] of cars with [speed > 0]"
"min" 1.0 0 -1184463 true "" "plot min [reward] of cars with [speed > 0]"

MONITOR
345
690
450
735
total reward
sum [reward] of cars with [speed > 0]
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
880
690
995
735
% accelerated action
(count cars with [ action = 0 ]) / nb-cars * 100
2
1
11

MONITOR
1000
690
1115
735
% decelerated action
(count cars with [ action = 2 ]) / nb-cars * 100
2
1
11

MONITOR
1120
690
1230
735
% stay-same action
(count cars with [ action = 1 ]) / nb-cars * 100
2
1
11

MONITOR
1235
690
1355
735
% change-lane action
(count cars with [ action = 3 ]) / nb-cars * 100
2
1
11

MONITOR
1580
295
1695
340
damaged cars
count cars with [speed = 0]
2
1
11

SLIDER
5
290
125
323
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
370
175
403
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
325
125
358
input-exp?
input-exp?
0
1
-1000

SWITCH
130
325
245
358
input-time?
input-time?
0
1
-1000

SLIDER
130
150
245
183
max-damaged-cars
max-damaged-cars
0
10
0.0
1
1
NIL
HORIZONTAL

SLIDER
5
660
245
693
simulation-time
simulation-time
10000
1000000
100000.0
1000
1
NIL
HORIZONTAL

TEXTBOX
5
425
155
443
Parameters for A2C
14
0.0
1

SWITCH
5
445
175
478
multiple-workers?
multiple-workers?
1
1
-1000

TEXTBOX
5
495
155
513
Parameters for PPO
14
0.0
1

SLIDER
5
515
175
548
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
570
175
603
dynamic-situation?
dynamic-situation?
0
1
-1000

SWITCH
5
610
175
643
greedy-attitude?
greedy-attitude?
0
1
-1000

SLIDER
130
290
245
323
n-step
n-step
1
10
3.0
1
1
NIL
HORIZONTAL

PLOT
1485
345
1875
515
Number of Collisions
Time
# Collisions
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"collisions" 1.0 0 -2674135 true "" "plot nb-collisions"

MONITOR
1485
295
1570
340
# collisions
nb-collisions
17
1
11

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
<experiments>
  <experiment name="marl_experiment_greedy" repetitions="15" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>mean [ speed ] of cars with [speed &gt; 0]</metric>
    <metric>mean [ patience ] of cars</metric>
    <enumeratedValueSet variable="move-strategy">
      <value value="&quot;Greedy&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-cars">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-lanes-oneway">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="acceleration">
      <value value="0.005"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="deceleration">
      <value value="0.02"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-patience">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-damaged-cars">
      <value value="4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-decay">
      <value value="0.001"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="learning-rate">
      <value value="0.0034"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discount-factor">
      <value value="0.98"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="batch-size">
      <value value="128"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-exp?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-time?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="memory-size">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="multiple-workers?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="clip_param">
      <value value="0.2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-situation?">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="marl_experiment_qlearning" repetitions="15" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>mean [ speed ] of cars with [speed &gt; 0]</metric>
    <metric>mean [ patience ] of cars</metric>
    <enumeratedValueSet variable="move-strategy">
      <value value="&quot;Q-Learning&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-cars">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-lanes-oneway">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="acceleration">
      <value value="0.005"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="deceleration">
      <value value="0.02"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-patience">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-damaged-cars">
      <value value="4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-decay">
      <value value="0.001"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="learning-rate">
      <value value="0.0034"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discount-factor">
      <value value="0.98"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="batch-size">
      <value value="128"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-exp?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-time?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="memory-size">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="multiple-workers?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="clip_param">
      <value value="0.2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-situation?">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="marl_experiment_greedy_speed" repetitions="5" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>mean [ speed ] of cars with [speed &gt; 0]</metric>
    <enumeratedValueSet variable="move-strategy">
      <value value="&quot;Greedy&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-cars">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-lanes-oneway">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="acceleration">
      <value value="0.005"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="deceleration">
      <value value="0.02"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-patience">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-damaged-cars">
      <value value="4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-decay">
      <value value="0.001"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="learning-rate">
      <value value="0.0034"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discount-factor">
      <value value="0.98"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="batch-size">
      <value value="128"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-exp?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-time?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="memory-size">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="multiple-workers?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="clip_param">
      <value value="0.2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-situation?">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="qlearning_speed&amp;reward_decay_1E-3" repetitions="10" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>mean [ speed ] of cars with [speed &gt; 0]</metric>
    <metric>sum [reward] of cars with [speed &gt; 0]</metric>
    <enumeratedValueSet variable="move-strategy">
      <value value="&quot;Q-Learning&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-cars">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-lanes-oneway">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="acceleration">
      <value value="0.005"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="deceleration">
      <value value="0.02"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-patience">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-damaged-cars">
      <value value="4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-decay">
      <value value="0.001"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="learning-rate">
      <value value="0.0034"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discount-factor">
      <value value="0.98"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="batch-size">
      <value value="128"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-exp?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-time?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="memory-size">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="multiple-workers?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="clip_param">
      <value value="0.2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-situation?">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="marl_experiment_sarsa_speed&amp;reward" repetitions="10" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>mean [ speed ] of cars with [speed &gt; 0]</metric>
    <metric>sum [reward] of cars with [speed &gt; 0]</metric>
    <enumeratedValueSet variable="move-strategy">
      <value value="&quot;SARSA&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-cars">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-lanes-oneway">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="acceleration">
      <value value="0.005"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="deceleration">
      <value value="0.02"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-patience">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-damaged-cars">
      <value value="4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-decay">
      <value value="0.001"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="learning-rate">
      <value value="0.0034"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discount-factor">
      <value value="0.98"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="batch-size">
      <value value="128"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-exp?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-time?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="memory-size">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="multiple-workers?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="clip_param">
      <value value="0.2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-situation?">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="marl_experiment_doubleql_speed&amp;reward" repetitions="10" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>mean [ speed ] of cars with [speed &gt; 0]</metric>
    <metric>sum [reward] of cars with [speed &gt; 0]</metric>
    <enumeratedValueSet variable="move-strategy">
      <value value="&quot;Double Q-Learning&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-cars">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-lanes-oneway">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="acceleration">
      <value value="0.005"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="deceleration">
      <value value="0.02"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-patience">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-damaged-cars">
      <value value="4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-decay">
      <value value="0.001"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="learning-rate">
      <value value="0.0034"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discount-factor">
      <value value="0.98"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="batch-size">
      <value value="128"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-exp?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-time?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="memory-size">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="multiple-workers?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="clip_param">
      <value value="0.2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-situation?">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="marl_experiment_qlgr_speed&amp;reward" repetitions="10" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>mean [ speed ] of cars with [speed &gt; 0]</metric>
    <metric>sum [reward] of cars with [speed &gt; 0]</metric>
    <enumeratedValueSet variable="move-strategy">
      <value value="&quot;Q-Learning&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-cars">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-lanes-oneway">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="acceleration">
      <value value="0.005"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="deceleration">
      <value value="0.02"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-patience">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-damaged-cars">
      <value value="4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-decay">
      <value value="0.001"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="learning-rate">
      <value value="0.0034"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discount-factor">
      <value value="0.98"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="batch-size">
      <value value="128"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-exp?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-time?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="memory-size">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="multiple-workers?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="clip_param">
      <value value="0.2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-situation?">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="marl_experiment_ssgr_speed" repetitions="5" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>mean [ speed ] of cars with [speed &gt; 0]</metric>
    <enumeratedValueSet variable="move-strategy">
      <value value="&quot;SARSA&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-cars">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-lanes-oneway">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="acceleration">
      <value value="0.005"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="deceleration">
      <value value="0.02"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-patience">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-damaged-cars">
      <value value="4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-decay">
      <value value="0.001"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="learning-rate">
      <value value="0.0034"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discount-factor">
      <value value="0.98"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="batch-size">
      <value value="128"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-exp?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-time?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="memory-size">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="multiple-workers?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="clip_param">
      <value value="0.2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-situation?">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="marl_experiment_dqlgr_speed&amp;reward" repetitions="10" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>mean [ speed ] of cars with [speed &gt; 0]</metric>
    <metric>sum [reward] of cars with [speed &gt; 0]</metric>
    <enumeratedValueSet variable="move-strategy">
      <value value="&quot;Double Q-Learning&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-cars">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-lanes-oneway">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="acceleration">
      <value value="0.005"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="deceleration">
      <value value="0.02"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-patience">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-damaged-cars">
      <value value="4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-decay">
      <value value="0.001"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="learning-rate">
      <value value="0.0034"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discount-factor">
      <value value="0.98"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="batch-size">
      <value value="128"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-exp?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-time?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="memory-size">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="multiple-workers?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="clip_param">
      <value value="0.2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-situation?">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="qlearning_speed&amp;reward_decay_1E-4" repetitions="10" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>mean [ speed ] of cars with [speed &gt; 0]</metric>
    <metric>sum [reward] of cars with [speed &gt; 0]</metric>
    <enumeratedValueSet variable="move-strategy">
      <value value="&quot;Q-Learning&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-cars">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-lanes-oneway">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="acceleration">
      <value value="0.005"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="deceleration">
      <value value="0.02"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-patience">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-damaged-cars">
      <value value="4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-decay">
      <value value="1.0E-4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="learning-rate">
      <value value="0.0034"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discount-factor">
      <value value="0.98"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="batch-size">
      <value value="128"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-exp?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-time?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="memory-size">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="multiple-workers?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="clip_param">
      <value value="0.2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-situation?">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="qlearning_speed&amp;reward_decay_5E-5" repetitions="10" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>mean [ speed ] of cars with [speed &gt; 0]</metric>
    <metric>sum [reward] of cars with [speed &gt; 0]</metric>
    <enumeratedValueSet variable="move-strategy">
      <value value="&quot;Q-Learning&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-cars">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-lanes-oneway">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="acceleration">
      <value value="0.005"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="deceleration">
      <value value="0.02"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-patience">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-damaged-cars">
      <value value="4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-decay">
      <value value="5.0E-5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="learning-rate">
      <value value="0.0034"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discount-factor">
      <value value="0.98"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="batch-size">
      <value value="128"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-exp?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-time?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="memory-size">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="multiple-workers?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="clip_param">
      <value value="0.2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-situation?">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="qlearning_speed&amp;reward_decay_0" repetitions="10" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>mean [ speed ] of cars with [speed &gt; 0]</metric>
    <metric>sum [reward] of cars with [speed &gt; 0]</metric>
    <enumeratedValueSet variable="move-strategy">
      <value value="&quot;Q-Learning&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-cars">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-lanes-oneway">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="acceleration">
      <value value="0.005"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="deceleration">
      <value value="0.02"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-patience">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-damaged-cars">
      <value value="4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-decay">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="learning-rate">
      <value value="0.0034"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discount-factor">
      <value value="0.98"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="batch-size">
      <value value="128"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-exp?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-time?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="memory-size">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="multiple-workers?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="clip_param">
      <value value="0.2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-situation?">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="qlearning_speed&amp;reward_decay_1E-4_staticsituation" repetitions="10" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>mean [ speed ] of cars with [speed &gt; 0]</metric>
    <metric>sum [reward] of cars with [speed &gt; 0]</metric>
    <enumeratedValueSet variable="move-strategy">
      <value value="&quot;Q-Learning&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-cars">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-lanes-oneway">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="acceleration">
      <value value="0.005"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="deceleration">
      <value value="0.02"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-patience">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-damaged-cars">
      <value value="4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-decay">
      <value value="1.0E-4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="learning-rate">
      <value value="0.0034"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discount-factor">
      <value value="0.98"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="batch-size">
      <value value="128"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-exp?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-time?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="memory-size">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="multiple-workers?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="clip_param">
      <value value="0.2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-situation?">
      <value value="false"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="qlgr_speed&amp;reward_decay_1E-4_staticsituation" repetitions="10" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>mean [ speed ] of cars with [speed &gt; 0]</metric>
    <metric>sum [reward] of cars with [speed &gt; 0]</metric>
    <enumeratedValueSet variable="move-strategy">
      <value value="&quot;Q-Learning&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-cars">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-lanes-oneway">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="acceleration">
      <value value="0.005"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="deceleration">
      <value value="0.02"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-patience">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-damaged-cars">
      <value value="4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-decay">
      <value value="1.0E-4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="learning-rate">
      <value value="0.0034"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discount-factor">
      <value value="0.98"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="batch-size">
      <value value="128"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-exp?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-time?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="memory-size">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="multiple-workers?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="clip_param">
      <value value="0.2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-situation?">
      <value value="false"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="qlearning_speed&amp;reward_decay_1E-5" repetitions="10" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>mean [ speed ] of cars with [speed &gt; 0]</metric>
    <metric>sum [reward] of cars with [speed &gt; 0]</metric>
    <enumeratedValueSet variable="move-strategy">
      <value value="&quot;Q-Learning&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="simulation-time">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-cars">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nb-lanes-oneway">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="acceleration">
      <value value="0.005"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="deceleration">
      <value value="0.02"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-patience">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-damaged-cars">
      <value value="4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="exp-decay">
      <value value="1.0E-5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="learning-rate">
      <value value="0.0034"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discount-factor">
      <value value="0.98"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="batch-size">
      <value value="128"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-exp?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="input-time?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="memory-size">
      <value value="100000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="multiple-workers?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="clip_param">
      <value value="0.2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-situation?">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
</experiments>
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
