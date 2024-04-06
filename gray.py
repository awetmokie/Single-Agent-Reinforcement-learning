from matplotlib import pyplot as plt
import gym
import highway_env
import numpy as np




screen_width, screen_height = 84, 84
config = {
    "offscreen_rendering":True,
    "observation": {
        "type":"GrayscaleObservation",
        "weights": [0.2989, 0.5870, 0.1140],
        "stack_size": 4,"observation_shape": (screen_width, screen_height),
    },
    "screen_width": screen_width,
    "screen_height": screen_height,
    "scaling": 1.75,
    "policy_frequency": 2
}

env = gym.make('joker-intersection-v0')
env.configure(config)
obs = env.reset()
print(obs.shape)


# _, axes = plt.subplots(ncols=4, figsize=(12, 5))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(obs[..., i], cmap=plt.get_cmap('gray'))
# plt.show()



