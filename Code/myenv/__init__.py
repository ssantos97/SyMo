
from gym.envs.registration import register
from Code.myenv import acrobot, pendulum, cartpole
register(
    id='MyCartPole-v0',
    entry_point='Code.myenv.cartpole:CartPoleEnv',
)

register(
    id='MyAcrobot-v0',
    entry_point='Code.myenv.acrobot:AcrobotEnv',
)

register(
    id='MyPendulum-v0',
    entry_point='Code.myenv.pendulum:PendulumEnv',
)


