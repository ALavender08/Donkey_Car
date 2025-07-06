**目前 GitHub 上只有 PPO 的訓練模型與測試獎勵紀錄，其餘壓縮檔請見雲端資料夾**
## PPO2
```python
PPO("CnnPolicy", env, n_steps=500, verbose=0, learning_rate=0.0003*0.1 )
```

## PPO3
```python
PPO("CnnPolicy", env, n_steps=500, verbose=0, learning_rate=0.0003*0.5 )
```

## PPO4
```python
PPO("CnnPolicy", env, n_steps=500, verbose=0 )
(default) learning_rate = 0.0003
```

## SAC
```python
SAC("CnnPolicy", env, n_steps=500, verbose=0, buffer_size = 10000 )
(default) learning_rate = 0.0003
```

## SAC2
```python
SAC("CnnPolicy", env, n_steps=500, verbose=2, buffer_size = 10000 )
(default) learning_rate = 0.0003
```

## SAC3
```python
SAC("CnnPolicy", env, n_steps=500, verbose=0, buffer_size = 10000, learning_rate = 0.00000005 )
```

## TD3
```python
class CustomTD3Policy(TD3Policy):
    def make_actor_critic_optimizer(self):
        self.actor.optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.00001 )
        self.critic.optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
```
```python
n_actions = env.action_space.shape[-1]
action_noise = stable_baselines3.common.noise.NormalActionNoise(mean=0.1*np.ones(n_actions), sigma=0.1 * np.ones(n_actions))
model = TD3(CustomTD3Policy, env, verbose = 0, buffer_size=10000, action_noise = action_noise )
```
有客製化學習率 + 加入雜訊

## TD32
```python
TD3("CnnPolicy", env, verbose = 0, buffer_size=10000, learning_rate = 0.00000005 )
```

## TD33
```python
class CustomTD3Policy(TD3Policy):
    def make_actor_critic_optimizer(self):
        self.actor.optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0000005 )
        self.critic.optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.000001)
```
```python
n_actions = env.action_space.shape[-1]
action_noise = stable_baselines3.common.noise.NormalActionNoise(mean=0.1*np.ones(n_actions), sigma=0.1 * np.ones(n_actions))
model = TD3(CustomTD3Policy, env, verbose = 0, buffer_size=10000, action_noise = action_noise )
```
有客製化學習率 + 加入雜訊
