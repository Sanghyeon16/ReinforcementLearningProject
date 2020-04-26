# Training only with throwing reward function
python train.py --env OurHumanoidThrow-v0 --model ppo --num-envs 8 --lr 3e-4 --num-env-steps 10000000 --use-linear-lr-decay --num-steps 256 --model-hiddens 500 500 --model-activation relu --no-visualize --save-dir "experiments/ppo-throw"

# Training in 3 stages
python train.py --env OurHumanoidStand-v0                                      --model ppo --num-envs 8 --lr 3e-4 --num-env-steps 3000000 --use-linear-lr-decay --num-steps 256 --model-hiddens 500 500 --model-activation relu --no-visualize --save-dir "experiments/ppo-stand"
python train.py --env OurHumanoidStandToHold-v0 --eval-env OurHumanoidHold-v0  --model ppo --num-envs 8 --lr 3e-4 --num-env-steps 3000000 --use-linear-lr-decay --num-steps 256 --model-hiddens 500 500 --model-activation relu --no-visualize --save-dir "experiments/ppo-stand-to-hold" --load-model "experiments/ppo-stand/model-3000000.pth"
python train.py --env OurHumanoidHoldToThrow-v0 --eval-env OurHumanoidThrow-v0 --model ppo --num-envs 8 --lr 3e-4 --num-env-steps 4000000 --use-linear-lr-decay --num-steps 256 --model-hiddens 500 500 --model-activation relu --no-visualize --save-dir "experiments/ppo-hold-to-throw" --load-model "experiments/ppo-stand-to-hold/model-3000000.pth"

