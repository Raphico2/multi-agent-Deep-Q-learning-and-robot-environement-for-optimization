multi-agent-Deep-Q-learning-and-robot-environement-for-optimization

The code refer to the full integration of a restaurant environment, managed by robots using multi-agent deep Q learning model 

Our project deals with robot cognition and collaboration in order to perform optmization task. The goal was to create a restaurant environment and train robots to manage and optmize the gestion of the restaurant. The restaurant is a Pizza fast food where customers enter, order, wait until be served and then pay and eat. The robots are divided into two type, waiter robots and cooker robots. They must learn how to cooperate to answer the demand as fast as possible, keep the restaurant clean and then improve the customer satsfaction in order to increase the income. The report aims to detail the environment, the robots, the AI model we used and our results.


# 1. The goal and environment

# Environment

The environment is a restaurant that can host 200 customers maximum a day. The restaurant is open for 4 hours which represent in our model 240 timestamp. Each day, every customer can decide to come or not, the probability of coming into the restaurant is set following a Bernoulli Law where the probability is the customer self-satisfaction. At the beginning, each customer has a probability of 0.5 to come, then, in funcGon of its last experience in the restaurant, its self-satisfaction can increase or decrease and then, influence the chances to return the next day. When a customer decides to come, its arrival timestamp is set following a normal law with a mean of 120 timmestamp and a variance of 60 timestamp.
The environment is partially observable for the robots who manage the restaurant because they don’t know customer satisfaction, the number of customer who will come and customer arrival Gme. Each day, a scheduler is built and program at what Gme customer will arrive, this information is not accessible for robots.
On the other side, robots must manage the restaurant properly in order to answer the demand as fast as possible. The robots can be either waiter or cooker. The cooker robot has 3 possible actions, cook a pizza, wash a plate, Recharge itself. Each of those actions take exactly one timestamp. The quantity of energy used by the robot for cooking or washing is different and the quantity of energy provided by 1 timestamp of charge is a parameter of our model. Waiter robot can take the order of a customer, deliver the pizza, clean the restaurant, and recharge itself. The quantity of robots of each type can be set as parameter of the model. Each action takes one timestamp.
Each customer enters the restaurant at a random timestamp, then its satisfaction is computed by calculating the time he waited from its arrival until being served. If the customer waits more than the average waiting time, the satisfaction is decreasing linearly timestamp after timestamp, if he doesn’t get served until the end of the day its satisfaction is set to 0 and he will never come back. Moreover, the self-satisfaction also depends on the cleanness level of the restaurant, each time a robot cooks or a waiter delivers, he let a small amount of dirt in the restaurant, the dirt quantity gives a cleanness score that is multiplied by the waiting score to give the final self-satisfaction score for a customer.
Each time a customer gets served, he pays 10$, each time a cooker robot makes a pizza, it cost 2$. (The price, cost, average waiting time and dirt score level can be set in the configuration of the model).
A simulation in our environment represents a year simulation, where the number of day and hours of activity per day can be set in the configuration of our environment. For the project, we trained our model on the basis of 4 hours opening time (240 timestamp) per days and a “year” of 50 days.


# Agents


Our agents are the robots, we want to train our agent to manage the restaurant as best as possible. Every robot has a type (cooker or waiter) and pre-defined actions it can perform as we described above. It is important to notice that every action take one timestamp which represents 1 minutes. The action of cook gives as result 1 pizza created. On the other side, there is a limited number of plate that need to be washed to be reused, when the cooker robot decides to perform the action “Wash”, a defined number of plate are washed in one timestamp, this number is a parameter that can be set in the configuration. It is the same for the action “Clean” of the waiter, the amount of dirt the robot is able to clean in one timestamp can also be set in the configuration of the environment.
Our model has been trained with 3 cooker robots and 3 waiters robots but can be change anytime


# Goal and expectation

Our goal project is to create a model that will teach our robots to manage the restaurant. The goal of our robots is to maximize the year income of the restaurant. In order to do it, robots have to collaborate to increase as best as possible the customer satisfaction, more a customer is satisfied, more the chance of coming back the next day is high and then, more the daily income will increase. In our project we will use reinforcement learning and implement a multi-agent Deep Q learning algorithm in order to teach the robots. We expect to find the best policy for our robots to manage the restaurant and reach the maximum possible daily income. Moreover, we want our robots to maximize the customer satisfaction and keep the restaurant clean.


# 2. Our code, model, and algorithm

# Code

Our code is divided into 9 python files. 6 of the python files aims to describe the environment, agent, visual render, and funcGons such as actions, pre-conditions, and effects. 1 python file is the reinforcement learning algorithm implementation, 1 is for training our model and the last one is for showing results.

- Config.py : Is a class where we can set all the parameters of the environment. We can set all actions efficiency, action energy cost, maximum number of customers, number hours of opening, average waiting time ... The config file must be instantiated to start the model, it can also allow the user to set the real variables of its restaurant.


- State.py : The file help to maintain the fluent of our model and create a numerical representation for each robot state of the word representation. In function of the time of robot (cooker or waiter) the representation of the environment is different. For cooker robots, the state representation is the level of battery of the robot, the number of available plate (cleaned) and the number of ordered pizza that have not been cooked yet. On the other side, the waiter robot has a different state representation which contains the number of accumulated dirt in the restaurant, the bahery level, for each customer the status and the waiting time.


- Action.py: The file describes the action space of the robots. It implements the post conditions and effect of an action, determines what action are possible from a given state for each robots and update the fluent once an action is performed. An action can be done if the current state allows it for a robot, once the action has been done, it updates the state and create a new one. Moreover, because there are many robots, the file has a lock function that can allow robots to lock resources at each timestamp in order to avoid duplicate utilization of resources and then to forbid a robot to perform the same action of another one. In our environment, 2 waiter robots cannot serve the same customer, or two cooker robot cannot prepare the same order.


- ModelizaGon.py: The file constructs the visual model render of our environment. It is represented like that at each timestamp. At each timestamp there are 2 representation of the environment. The first one without the actions chosen and the second one with the actions chosen. In Yellow the cooker robots, in orange the waiter robots. On the left side of the dashboard the bahery level of each robots. On the right side of the dashboard, all the customers present in the restaurant are represented. When a customer is in pink, It means he has not ordered yet, when he is in green it means he is waiting to be delivered. The number written inside the customer is its waiting time since its arrival. On the right side of the blue rectangle, the number of accumulated dirt, delivered customer, available plate, Pizza ordered and Pizza waiting to be served are written.
At each timestamp, robots are choosing an action to perform, this action appears on the dashboard at each timestamp in a grey rectangle.


- Environement.py : It is the global implementation of the environment, it coordinates the state file, the actoon file, the visual model file and the config file. It is implemented as a gym environment with the functions step, render and reset. The function step simulates a timestamp from the moment each robots get their action to perform. The environment class allow the simulator to get all the information of the current global state of the world but also the observaGon space of each robot at each timestamp and their action space and possible actions space. We can reset or start a new day from the environment class.


- UGls.py: Contains the function for getting the possible action with the greater Q value for a given robot and the function that determine randomly arrival time of each robot.


- DQN.py: The implementation of the multi-agent Deep Q Learning Algorithm for our robots. Each robot has a brain which is Fully connected neural network. The NN receives at each timestamp the current observation space of the robot and return the Q-Value for each actions in the robot action space. At each timestamp, the robot will take the action with the highest Q-Value among the actions that are possible to perform at this timestamp. The file also contains a function predict and learn. The function predicts take the current observation of each robot and return the Q-values for each robot. The function learn is a function that keeps in memory all the actions and reward a robot performed in a year and go backward in the NN of each robots to update the robot knowledge.


- Simulator.py: The Simulator is made for training the model during a long period of time. During 2000 simulated year containing 50 days each and 240 timestamp per days, robots are exploring and then exploiting their environment by creating new strategy and check their efficiency. We implemented and epsilon-greedy policy to make the algorithm explore at the beginning and then exploit more as far we get in the training procedure. The main purpose is to let the robot understand how to maximize the profit in a year. A[er each day, robots received a common reward which is the mean satisfaction of all the customers that ate in the restaurant during the day. Because we know that maximizing customer satisfaction will increase the number of customer the next day and then increase profits, we thought that this reward is the best way for robots to understand and reach their objective. If the customer satisfaction is high, the reward will be greater and then the robots will keep the best policy that maximize the customer satisfaction.


- TestSimulator.py: The file is the main file to see an example of a day in the model and perform a year simulation of the restaurant and see the result in term of day and year income. The file have to be used a[er the training to see if we did obtained good results in our project.


# Model and Algorithm

Because multi-agent reinforcement learning is a very challenging task, especially for robots, we decide to try it for our project. Our algorithm is a Deep Q learning fully decentralized. It means that each robot has its own Neural Network (in our case a fully connected neural network with 2 hidden layers) and each robot takes its own decision regarding its observation of the world. Each robot has a partial observation space of the restaurant. The goal of the model is to make the robot learn the best policy to maximize customer satisfaction. During the training procedure, the model function “predict” is called at each timestamp to give the Q-value for each action for each robot. The prediction can be random if the model is exploring the environment or can be done with the Neural Network if the model is exploiting the environment.
At each timestamp, the actions and the observation space for each robot are memorized. At the end of the day, robots are getting the reward (customer satisfaction mean) and then the weight of the Neural network of each robot is updated with backward propagation.
After the learning procedure, the brains of each robot are saved and then can be reused for testing the model.


# Reward and punishment


In our RL model we had to define a reward and punishment policy. We wanted at the beginning to set the daily income as reward but we figured out that it was not relevant because customers can pay even if they are not satisfied. We decided to set the global satisfaction as the final reward of the day. We also added small reward during a day to encourage or discourage robots to perform or not perform actions.
For the cooker robots we applied a penalty if they choose to not do anything if they are things to do, moreover a small penalty is set when the cooker robot decides to recharge if it has enough battery to perform other activity.
For the waiter robot, we give a reward each time the robot chooses to deliver or take the order of the customer who is first on the waiting list. The size of the reward increases if the robot decides to serve or take the order of someone who is on the top of the waiting list, and get close to 0 if the robot choose to serve someone who is far from the top. We also punish a waiter who is not doing anything if there are things to do and reward them when they clean the restaurant if the dirt level is high.


# Results


We trained our algorithm during 10000 simulated years or 325 days with 240 timestamp per day. The tradeoff between exploration and exploitation is decreasing over timestamp and reach a level of 10%-90% after the year 7000. The training time of our algorithm was very long, it took us 6 days to complete the training part. Each simulated year take approximatively 1 minutes. Afer the training part, we get the following results:

- The Robots succeeded to deal with an amount of 150 customers a day and get them all satisfied with a satisfaction score of 0.8.
- The daily income reach afer the year 8000 an amount between 1100$ and 1200$ (when 1200 is the maximum possible amount since there can be maximum 150 customers in the restaurant)
- The Restaurant is almost all the time above the first level of dirt.
- The robots succeeded to learn that they need all the time battery and go recharge themselves when there amount of bahery gets lower.
- Afer the year 4500 robots understood that in order to maximize the customer satisfaction, all the customers must be served after the end of the day.
- Afer the year 6350 robots understood that the best policy is to serve the customer which come first in the restaurant.

The training was really satisfying for the amount of 200 customers maximum for 6 robots in the restaurant. We also saw that Robots improved their coordination, for example, 2 of the cooker robots are almost all the time cooking where 1 is almost all the time washing the plate.
Afer training, we make a record of a day and a year of simulation with the visualization in order to show our results. Notice that we can show our results and modifying parameters directly on the code in the file simulator.py.
We also collected the results over a year and see how many customers the restaurant arrives to maintains. With our environment algorithm, more a customer is satisfied from its last visit, more the chances to come back the next day are high. For an amount of 200 customers maximum and a satisfaction rate of 0.5 at the beginning, the robots start with a number of customer approximatively equal to 100. They get to answer the demand for all customer to increase their satisfaction to get more customers the next day, then they have to again answer the demand. We noticed that when the number of customer in the restaurant is lower than 125, the customer satisfaction lay between 0.53 and 0.65 which is better than the beginning, we also noticed that the robots arrive to answer the demand for all customer, nobody is not served after the end of the service.
When the number of customers is higher than 125, robots have difficulties to maintain the satisfaction level above 0.5 and to serve all the customers.

After a year of service, the customer satisfaction lay between 0.55 and 0.6 and the average number of customer a day is 123. Moreover, the daily income of the restaurant is 915$ of profits which is a very good results.

Here is a link for showing a short video for showing the results :
https://youtu.be/e6wnH9OFvDk
Anyone can also run the code in the file simulator.py and see the results directly


# Limits and improvement


The results showed us that using reinforcement Learning for the gestion of a restaurant by robots can help improve results in the domain. Our results showed that this type of algorithm works and give to robots an intelligence that could help resolving complex tasks. However, we met some issues such as the fact that robots did not get to predict the probability law of customers arrival (normal law with mean 120 and variance 60). They don’t really act in function of their arrival but in function of which customer is here. Another issue is that robots seem to take order and deliver randomly to customer and not by their arrival time.
We think that more training and more exploration would help to improve results. We also have a limitation for the number of customers the robots get to handle. We see our limitation at 125 customers a day. We think that this limitation could be passed with more training or increasing the number of robots in the restaurant.
