# AI3603

## 简介

本仓库为 上海交通大学 **人工智能理论及应用(AI3603)** 课程平时作业，完成于2025年秋季学期，任务要求详见 [平时作业 1](https://github.com/PasserbyZzz/AI3603/blob/main/HW1/report/AI3603_HW1.pdf)、[平时作业 2](https://github.com/PasserbyZzz/AI3603/blob/main/HW2/report/AI3603_HW2.pdf) 和 [平时作业 3](https://github.com/PasserbyZzz/AI3603/blob/main/HW3/report/AI3603_HW3.pdf)。

## 总体流程

### 平时作业 1：搜索算法

在本作业中，你将为房间内的服务机器人使用 A* 算法开发一个路径规划框架。

### 平时作业 2：强化学习

在本作业中，你将实现使用强化学习算法的智能体。

### 平时作业 3：贝叶斯网络

在本作业中，你将使用变量消去法实现贝叶斯网络。

## 具体实现

详见 [平时作业 1 实验报告](https://github.com/PasserbyZzz/AI3603/blob/main/HW1/report/HW1_report.pdf)、[平时作业 2 实验报告](https://github.com/PasserbyZzz/AI3603/blob/main/HW2/report/HW2_report.pdf) 和 [平时作业 3 实验报告](https://github.com/PasserbyZzz/AI3603/blob/main/HW3/report/HW3_report.pdf)。

## 效果展示

### 平时作业 1

<p align="center">
  <table align="center">
    <tr>
      <td align="center">
        <img src="HW1\figures\Task_1.png" 
             width="400" 
             alt="Task 1">
        <br>
      </td>
      <td align="center">
        <img src="HW1\figures\Task_2.png" 
             width="400" 
             alt="Task 2">
        <br>
      </td>
      <td align="center">
        <img src="HW1\figures\Task_3_safetybuffer_3.5.png" 
             width="400" 
             alt="Task 3">
        <br>
      </td>
    </tr>
  </table>
</p>

### 平时作业 2

<p align="center">
  <table align="center">
    <tr>
      <td align="center">
        <img src="HW2\videos\qlearning-cliffwalk.gif" 
             width="400" 
             alt="qlearning-cliffwalk">
        <br>
      </td>
      <td align="center">
        <img src="HW2\videos\dqn-lunarlander.gif" 
             width="200" 
             alt="dqn-lunarlander">
        <br>
      </td>
    </tr>
  </table>
</p>

### 平时作业 3

<div align="center">
  <img src="HW3\screenshots\income_effect_plot.png" 
       alt="income_effect_plot" 
       style="width: 80%; max-width: 800px;">
</div>

## 文件目录

- **AI3603**
  - **HW1**：平时作业 1：搜索算法
    - **map**：地图文件
    - **report**：平时作业 1 要求与实验报告
    - **`Task_1.py`**：四连通 A* 算法
    - **`Task_2.py`**：八连通 A* 算法
    - **`Task_3.py`**：混合 A* 算法
  - **HW2**：平时作业 2：强化学习
    - **data**：SARSA、Dyna-Q 和 Q-Learning 模型
    - **models**：DQN 模型
    - **report**：平时作业 2 要求与实验报告
    - **runs**：DQN 训练过程记录
    - **`agent.py`**：SARSA、Dyna-Q 和 Q-Learning 算法实现
    - **`cliff_walk_dyna_q`**：Dyna-Q 训练实现
    - **`cliff_walk_qlearning`**：Q-Learning 训练实现
    - **`cliff_walk_sarsa`**：SARSA 训练实现
    - **`dqn`**：DQN 算法、训练实现，Lunar Lander 环境视频生成
    - **`evaluate_cliffwalk`**：Cliff Walk 环境评估
    - **`video_cliffwalk`**：Cliff Walk 环境视频生成
  - **HW3**：平时作业 3：贝叶斯网络
    - **report**：平时作业 3 要求与实验报告
    - **`BayesianNetworks.py`**：贝叶斯网络实现
    - **`BayesNetworkTestScript.py`**：贝叶斯网络测试脚本
    - **`RiskFactorsData.csv`**：风险因素数据
  - **Lab1**：使用 A* 算法实现 HTML5 canvas 版本的鼠标位置追踪

## 邮箱

任何疑问，欢迎邮件交流：**`passerby_zzz@sjtu.edu.cn`** !

## **Wish for your Star⭐!**