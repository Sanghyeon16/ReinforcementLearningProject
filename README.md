# Learning to Throw with Reinforcement Learning
In this project, we intended to train a humanoid robot to throw a ball using only reinforcement learning. The goal is to let the robot learn to throw a ball as fast or as far as possible.

## Requirements
* [OpenAI gym](https://github.com/openai/gym)
* [OpenAI baselines](https://github.com/openai/baselines)
* [Pytorch](https://pytorch.org/)

## Usage
To train a model, use the following command.
```
python train.py [options for training] [options for specific algorithm]
```
During training, the process saves the snapshot of the model in the file `model-<num_steps>.pth`, where `<num_steps>` is the total number of training steps.  The model is saved in the directory defined in `--save-dir`. To retrain a saved model, use the option `--load-model` to reload the `.pth` file.


To evaluate the saved model on a specific environment use the following command.
```
python evaluate.py <saved_model.pth> [options for evaluation]
```
It will save the video of testing in the directory defined in `--save-dir`

### Environment names
We implemented five environments for our humanoid to learn.

* OurHumanoidStand-v0
* OurHumanoidHold-v0
* OurHumanoidThrow-v0
* OurHumanoidStandToHold-v0
* OurHumanoidHoldToThrow-v0
 
### Options for training
|Options | Descriptions |
|---|---|
|`--env {env_name}`| The environment for training.|
|`--model {"ddpg" or "ppo"}`| The training algorithm.|
|`--num-envs {number}`| The number of training environments running in parallel.|
|`--save-dir {dir}` | The directory to save snapshots of model and logging files.|
|`--load-model {model.pth}`| Retrain the saved snapshot of model.|

### Options for evaluation
|Options | Descriptions |
|---|---|
|`--env {env_name}`| The environment for testing.|
|`--num-eval {number}`| The number of times to perform evaluation.|
|`--save-dir {dir}` | The directory to save the video of running testing.|

### Options for PPO
|Options | Descriptions |
|---|---|
|`--lr {float number}`| The learning rate for training.|
|`--use-linear-lr-decay`| With this option, the learning rate will decay linearly to 0 at the end of training.|
|`--model-hiddens {numbers, e.g. 500 500}`| The dimension of hidden layers.|
|`--model-activation {"tanh" or "relu"}` | Use tanh or ReLU as activation function.|
|`--num-steps {number}`| The number of steps is in a training episode.|
|`--num-env-steps`|The total number of training steps.|
|`--save-interval`| How many steps between saving the snapshots of the model.|

### Options for DDPG
|Options| Descriptions|
|---|---|
|`--rate {float number}`| The learning rate.|
|`--prate {float number}`| The learning rate for policy.|

## Examples
* Train to stand stably using PPO
```
python train.py --env OurHumanoidStand-v0 --model ppo --num-envs 8 --lr 3e-4 --num-env-steps 10000000 --use-linear-lr-decay --num-steps 256 --model-hiddens 500 500 --model-activation relu --save-dir "experiments/ppo"
```


## Codes borrowed from
* [OpenAI gym](https://github.com/openai/gym)
* [dm_control](https://github.com/deepmind/dm_control)
* [DDPG](https://github.com/ghliu/pytorch-ddpg)
* [PPO](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)
