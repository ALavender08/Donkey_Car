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
有客製化學習率 + 加入雜訊

## TD32
```python
TD3("CnnPolicy", env, verbose = 0, buffer_size=10000, learning_rate = 0.00000005 )
```

## TD33
有客製化學習率 + 加入雜訊
