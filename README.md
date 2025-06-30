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
