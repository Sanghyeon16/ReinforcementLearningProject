#python train.py --env OurHumanoidStand-v0                                      --model ppo --num-envs 8 --lr 3e-4 --num-env-steps 3000000 --num-steps 256 --model-hiddens 500 500 --model-activation relu --no-visualize --save-dir "experiments/ppo-stand-4"

#python train.py --env OurHumanoidStandToHold-v0 --eval-env OurHumanoidHold-v0  --model ppo --num-envs 8 --lr 3e-4 --num-env-steps 3000000 --num-steps 256 --model-hiddens 500 500 --model-activation relu --no-visualize --save-dir "experiments/ppo-stand-to-hold-4" --load-model "experiments/ppo-stand-4/model-3000000.pth"

#python train.py --env OurHumanoidHoldToThrow-v0 --eval-env OurHumanoidThrow-v0 --model ppo --num-envs 8 --lr 3e-4 --num-env-steps 4000000 --use-linear-lr-decay --num-steps 256 --model-hiddens 500 500 --model-activation relu --no-visualize --save-dir "experiments/ppo-hold-to-throw-4" --load-model "experiments/ppo-stand-to-hold-4/model-3000000.pth"

#python train.py --env OurHumanoidThrow-v0 --model ppo --num-envs 8 --lr 3e-4 --num-env-steps 10000000 --use-linear-lr-decay --num-steps 256 --model-hiddens 500 500 --model-activation relu --no-visualize --save-dir "experiments/ppo-throw-1"
