# DonkeyCar之強化學習訓練

## 成果摘要
- 本次研究利用強化學習的連續動作演算法訓練 DonkeyCar 環境，於賽道上跑越遠代表分數越高、成效越好，最終報告呈現  PPO 演算法在不同學習率下的訓練成效。從結果可以看出，強化學習演算法較難訓練，學習率稍大便易有明顯分數震盪。
- 原先預期訓練出 TD3、SAC 模型並相互比較各演算法表現，但由於硬體設備的空間限制，沒能完成這部分研究。根據 PPO 的實作成果，推測上述兩種演算法的學習率有可能改進空間；從 SAC 的分數趨勢圖中，推測其可能需更多回合訓練才有更穩定的分數表現；從參考資料中，推測分數機制的修改也是可研究改進的方向之一。

## 動機
科技專題課程的邱崑山老師在高二下學期末提供數堂課讓我們進行強化學習的演算法實作，並推薦數個環境。我原先想選 Carla 環境，但由於其高規格硬體要球，使我改而選擇  DonkeyCar  環境。雖然此環境不如 Carla 具多樣化參數可供調整，但與科展的 Car Racing 環境相比，環境從原先的 2D 空照圖轉成 3D 圖像，演算法由離散動作改為連續動作的模型訓練，對我來說這是一項進步且有趣的一週研究。

## 事前準備
1. 按 [DonkeyCar](https://docs.donkeycar.com/) 官網步驟進行下載各套件、設定環境與測試
2. 參考 [stable baseline](https://stable-baselines3.readthedocs.io/en/master/) 官網下載模組
```
pip install stable-baselines3[extra]
```

## 研究方法
本次專題我選用 stable baselines3 搭建 PPO 模型，每一模型訓練至 100000 步或有能力完全跑完賽道後即終止訓練，其中我訓練了三種 PPO 模型，學習率分別為 0.0003、0.00015、0.00003。具體模型訓練架構搭建方法如下：
```python
for i in range(1000):
    # preparation
    if i == 0:
        model = PPO("CnnPolicy", env, n_steps=500, verbose=0, learning_rate=0.0003*0.5 )
        Log={"TestReward":[]}

    # training
    # 每 2500 步存檔一次
    obs = env.reset()
    model.learn(total_timesteps=2500, progress_bar=0)
    model.save(f'{folder}{(i+1)*2500}.pth')

    # testing
    # 測試 20 回合
    tem = 0
    for _ in range(20):
        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            # print (reward, done)
            env.render()
            tem += reward
            if done: break
    # 儲存每訓練 2500 回合訓練後測試 20 回合的平均獎勵
    Log["TestReward"].append(tem/20)
    print(f"testing {(i+1)*2500} : {tem/20}")
    np.save(f"{folder}/Log_PPO3-{(i+1)*2500}.npy", Log)
```

## 研究結果與分析
lr = 0.00003 的模型訓練到在 22500 步時即訓練完成（可連續跑 >10 圈），表現最好；其次為 lr = 0.00015 的模型，該模型在訓練 25000 步時的表現最好，可以完成賽道，但從趨勢圖可以看出其不穩定性；lr = 0.0003 模型的測試分數最低，也無法完成完整賽道。  
由以上推論，PPO 的學習率調的較小對訓練較有利，可能原因為強化學習演算法較難訓練，即使正在朝著正確的方向訓練，但學習率稍大便容易震盪，導致模型的策略越來越糟。  
[點擊此看 PPO 模型每訓練 2500 步便測試 20 回合的平均獎勵圖表](https://drive.google.com/file/d/1wZBe7owPa8z8y8uYGebTkyBaXkILQKzK/view?usp=sharing)

## 討論與心得
1. 從本次研究，更具體了解並實作 PPO 及其相關演算法
2. 從本次研究，更具體了解並實作 stable baselines3、git 等的使用方法
3. 從本次研究，將 PPO 演算法實作於 donkey-minimonaco-track-v0 環境中，並透過調整學習率，推測學習率於訓練模型的影響
4. 本次研究的 PPO 於學習率較大時較不穩定，除了降低學習率，調整獎勵或其他參數等方法也值得後續的探討

## 未來展望
在本次研究中，我曾使用 TD3、SAC 兩種演算法訓練模型，但 10 萬回合後其模型仍無法完整駛完一圈賽道。由於有調整過學習率使 PPO 模型更好先例，並發現在 stable baseline3 若要分別調整 Actor、Critic 的學習率可以客製化策略，這是可以投入更多研究去確認能否改進模型的一項變因。  
再者，從 SAC 的測試獎勵有逐步上升趨勢，推測兩演算法模型表現不如 PPO 的另一種可能原因為訓練回合不夠，而此兩種演算法的學習速度較慢，可能需要訓練至百萬回合才有能力穩定完成賽道。  
此外，獎勵也是影響訓練的一大因素。從[參考資料](https://towardsdatascience.com/suicidal-rl-agents-68159fc8f15a/)中得知，若 Agent 認為所得到的獎勵皆是負面的，可能會傾向學習自殺，以避免累積過多負面獎勵。而從另一[參考資料](https://www.reddit.com/r/reinforcementlearning/comments/1d8r7kz/my_td3_keep_suiciding_even_if_the_reward_is_worse/)中的討論得知可以試著修改環境，使獎勵乘上某個倍數，讓獎勵、懲罰的意義更明確，也能改進前一則資料所說的模型自毀行為，但實際的獎勵修改數值仍需後續實驗釐清。

## 附錄
- [研究報告](https://docs.google.com/document/d/1ZucsytjTjjlciN-wpHwn7taSAWhV_F93/edit?usp=sharing&ouid=116693472986107286060&rtpof=true&sd=true)
- [雲端資料夾（存放研究報告、分數折現圖、SAC & TD3模型、第一視角模型執行成果影片）](https://drive.google.com/drive/folders/1D9AbS5dj5ZhpvmzaS_2PAUA1tgqqT-5l?usp=drive_link)
- [研究時筆記](https://docs.google.com/document/d/156_-ESvBjmCfHAaZ1IflRwx6XCRnGf3u4RhfXAledQw/edit?pli=1&tab=t.xx4ufiyx847j#heading=h.am3mvux25hyk)
