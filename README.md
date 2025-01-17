# 参考编译：

https://blog.csdn.net/jiachang98/article/details/121700288

https://blog.csdn.net/ldk3679/article/details/105816907

加上gpt

终于编译成功了（ubuntu 22.04）



# 详解：

https://zhuanlan.zhihu.com/p/372956625







# 后端优化 backend

## 1 总体流程

典型的基于关键帧的 SLAM 后端优化流程：

**1. 初始化**

- 创建后端优化器对象 Backend，并启动一个后台线程 BackendLoop 用于执行优化任务。
- 后端线程会一直运行，等待前端发送优化请求。

**2. 接收优化请求**

- 当 SLAM 系统前端积累了一定数量的新关键帧和路标点后，会调用 Backend::UpdateMap() 方法，通知后端进行优化。



**3. 准备优化数据**  （获取关键帧和路标点）

- 后端线程收到优化请求后，会从地图中**获取需要参与优化的活动关键帧和路标点**。
- **活动关键帧**通常是指最近的关键帧，而**活动路标点**是指在多个关键帧中都能被观测到的路标点。



**4. 构造 g2o 优化问题**  （顶点：关键帧位姿，路标点坐标    边：投影模型）

- **创建优化器：**  实例化 g2o 优化器对象 **SparseOptimizer**，并选择合适的求解算法（例如 Levenberg-Marquardt）。

- **添加顶点：** 将**每个活动关键帧的位姿**表示为一个位姿顶点 VertexPose，将每个活动路标点的 3D 坐标表示为一个路标点顶点 VertexXYZ。

- **添加边（约束）：**  根据相机观测建立关键帧位姿与路标点位置之间的约束关系，每个约束表示为一条边 EdgeProjection。

  - 边的类型：  这里使用的是 EdgeProjection，表示相机投影模型，将 3D 点投影到 2D 图像平面上。
  - 边的信息矩阵：  反映了观测的精度，通常由特征点提取和匹配的协方差矩阵决定。
  - 鲁棒核函数：  用于处理可能存在的 outlier 观测，例如 Huber 核函数。

  

**5. 执行优化**

- 调用 optimizer.optimize() 方法执行优化，g2o 会根据图的结构和约束关系，迭代地调整关键帧位姿和路标点位置，使得整体误差最小化。



**6. 剔除 outlier**

- 优化完成后，检查每条边的误差，如果误差超过预设阈值，则认为该边对应的观测是 outlier，需要剔除。
- 剔除 outlier 可以提高地图和位姿估计的精度和鲁棒性。

**7. 更新地图**

- 将优化后的关键帧位姿和路标点位置更新到地图中，完成一次地图优化。

**循环**

- 后端线程会继续等待下一个优化请求，不断地对地图进行优化，提高 SLAM 系统的精度和一致性。

**总结**

这段代码展示了一个典型的基于关键帧和 g2o 的 SLAM 后端优化流程。通过不断地优化关键帧位姿和路标点位置，可以有效地减少 SLAM 系统的累积误差，构建更加精确的环境地图。

