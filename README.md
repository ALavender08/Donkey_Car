### 這裡是一個紀錄我做 donkey car 的強化學習專題的程式儲存庫

目前暫定要使用的演算法為：
- PPO
- TD3
- SAC
---
6/28 進度：
- 確認 donkey car 環境可由人工從網頁操控，且可以同時操縱 >1 台車
- 確認可以利用隨機動作操作 donkey car 環境中的車輛，且可以同時操縱 >1 台車
- 撰寫 PPO 程式（目前卡在用 stable baselines3 架模型，接下來打算確認是否需要改 Policy 參數）
- 學習使用 git/github
---
6/29 進度：
- 確認可以利用 PPO 演算法訓練模型，但後期會卡在原處不動，只有輪胎轉而向不加速
- 確認可以利用 TD3 演算法訓練模型，但前期就會卡在原處不動，只有輪胎轉向而不加速  
目前在研究[這部影片](https://www.youtube.com/watch?v=ngK33h00iBE)，52:38 開始似乎有解方（待驗證）
---
6/30 進度：
- 按照[影片](https://www.youtube.com/watch?v=ngK33h00iBE)步驟，改套用 rl baselines3 zoo 建模型  
（目前[影片](https://www.youtube.com/watch?v=ngK33h00iBE)進度到 42:55，準備改超參數）
---
7/1 進度：
- 套用 rl baselines3 zoo 建模型失敗，維持使用自己寫的程式訓練
- 參考[影片](https://www.youtube.com/watch?v=ngK33h00iBE) 00:51:36 ~ 01:00:30，修改環境  
（在 determine_episode_over (確認回合是否結束函式) 下新增以下程式，用以判斷車子是否停在一地或幾乎停住）
```python
if abs(self.speed) < self.min_speed and self.n_steps > 100:
    self.n_steps_low_speed +=1
    if self.n_steps_low_speed > 30:
	self.over = True
else:
    self.n_steps_low_speed = 0
```
- 自行改寫環境  
（在 observe (每一步的回傳函式) 下新增以下程式，在原本獎勵機制基礎上，給予慢速車懲罰與開很遠獎勵）
```python
reward = self.calc_reward(done) * 1.5
if self.speed < self.min_speed: 
    reward -= 3
reward += 0.002*self.n_steps
```
- 訓練 PPO 模型
---
7/2 進度：
- 訓練完成 PPO 模型（最好模型約在 60000 步時訓練好，後續模型表現又下降了）
- 嘗試訓練 TD3 模型（失敗：會立刻加速並撞牆）
- 嘗試訓練 SAC 模型（失敗：訓練很多回合還是無法跑完整圈）
---
7/3
- 由於訓練成效不佳，移除自定義的獎勵計算，但仍保留自定義終止條件
- 嘗試調整參數，改進 TD3、SAC（失敗，依舊會一直撞牆）
- 繪製邊訓練邊測試的獎勵圖表
- 撰寫報告（環境介紹）
- 訓練完成 PPO 模型（lr = 0.00003，最終模型確定可以持續跑在賽道上）
- 訓練完成 PPO 模型（lr = 0.00015，最終模型確定可以持續跑在賽道上）  
--> 由上兩模型推測，強化學習的學習率越小越容易訓練，若學習率太大很容易震盪
---
7/4
- 撰寫報告（全）
- 製作模型實際跑賽道的影片
- 嘗試調整參數，改進 TD3、SAC（失敗）
---
7/5
- 整理並上傳檔案至 GitHub、Google Drive
