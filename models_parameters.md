**PPO2 只留存 22500 回合的訓練模型（已訓練完成）**
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
