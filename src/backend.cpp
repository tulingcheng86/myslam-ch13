
#include "myslam/backend.h"
#include "myslam/algorithm.h"
#include "myslam/feature.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/mappoint.h"


// 一般编程中都需要先检查一个条件才进入等待环节，因此在中间有一个检查时段，检查条件的时候是不安全的，需要lock
//实现了 SLAM 系统中使用 g2o 进行后端优化的部分
namespace myslam {

Backend::Backend() {
    backend_running_.store(true);
    backend_thread_ = std::thread(std::bind(&Backend::BackendLoop, this));
}

// 触发地图优化
void Backend::UpdateMap() {
    std::unique_lock<std::mutex> lock(data_mutex_);
    map_update_.notify_one();
}

// 停止后端线程
void Backend::Stop() {
    backend_running_.store(false);
    map_update_.notify_one();
    backend_thread_.join();
}

// 后端优化循环（在单独的线程中运行）
void Backend::BackendLoop() {
    while (backend_running_.load()) {
        std::unique_lock<std::mutex> lock(data_mutex_);
        map_update_.wait(lock);// 等待优化信号
         // wait():一般编程中都需要先检查一个条件才进入等待环节，因此在中间有一个检查时段，检查条件的时候是不安全的，需要lock

        /// 后端仅优化激活的Frames和Landmarks
        // 收到信号后，使用活动的关键帧和路标点优化地图
        Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();
        Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();
        Optimize(active_kfs, active_landmarks);
    }
}

// 使用 g2o 的核心优化函数
void Backend::Optimize(Map::KeyframesType &keyframes,
                       Map::LandmarksType &landmarks) {
    // setup g2o
    //优化器构造可以参照： https://www.cnblogs.com/CV-life/p/10286037.html
    typedef g2o::BlockSolver_6_3 BlockSolverType;//// 位姿有 6 个自由度（3 个位置，3 个旋转）
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(
            std::make_unique<LinearSolverType>()));
    
    //创建稀疏优化器
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);// 设置求解器

    // 2. 添加相机位姿顶点
    // pose 顶点，使用Keyframe id
    std::map<unsigned long, VertexPose *> vertices;// 存储相机位姿顶点
    unsigned long max_kf_id = 0;
    for (auto &keyframe : keyframes) {// 遍历每个关键帧
        auto kf = keyframe.second;
        VertexPose *vertex_pose = new VertexPose();  // 创建相机位姿顶点
        vertex_pose->setId(kf->keyframe_id_);// 设置顶点 ID
        vertex_pose->setEstimate(kf->Pose());// 设置初始位姿估计值
        optimizer.addVertex(vertex_pose);// 将顶点添加到优化器
        if (kf->keyframe_id_ > max_kf_id) {
            max_kf_id = kf->keyframe_id_;
        }

        vertices.insert({kf->keyframe_id_, vertex_pose});// 保存顶点
    }


    // 3.添加路标顶点，使用路标id索引
    std::map<unsigned long, VertexXYZ *> vertices_landmarks;// 存储路标点顶点

    // 4. 相机参数 -- K 和左右外参
    Mat33 K = cam_left_->K();
    SE3 left_ext = cam_left_->pose();
    SE3 right_ext = cam_right_->pose();

    // edges
    // 5. 添加边 (观测约束)
    int index = 1;
    double chi2_th = 5.991;  // robust kernel 阈值
    std::map<EdgeProjection *, Feature::Ptr> edges_and_features;

    for (auto &landmark : landmarks) {// 遍历每个路标点
        if (landmark.second->is_outlier_) continue;
        unsigned long landmark_id = landmark.second->id_;
        auto observations = landmark.second->GetObs(); // 获取该路标点的所有观测
        for (auto &obs : observations) {// 遍历每个观测
            if (obs.lock() == nullptr) continue;
            auto feat = obs.lock();// 获取特征点
            if (feat->is_outlier_ || feat->frame_.lock() == nullptr) continue;

            auto frame = feat->frame_.lock();// 获取特征点所属的帧
            EdgeProjection *edge = nullptr;
            if (feat->is_on_left_image_) {
                edge = new EdgeProjection(K, left_ext);
            } else {
                edge = new EdgeProjection(K, right_ext);
            }

            //// 如果路标点landmark还没有对应的顶点，则创建一个新的顶点
            if (vertices_landmarks.find(landmark_id) ==
                vertices_landmarks.end()) {
                VertexXYZ *v = new VertexXYZ;
                v->setEstimate(landmark.second->Pos());
                v->setId(landmark_id + max_kf_id + 1);
                v->setMarginalized(true);
                vertices_landmarks.insert({landmark_id, v});
                optimizer.addVertex(v);
            }

            // 如果观测对应的关键帧和路标点都已经添加到了优化器中，则添加边
            if (vertices.find(frame->keyframe_id_) !=
                vertices.end() && 
                vertices_landmarks.find(landmark_id) !=
                vertices_landmarks.end()) {
                    edge->setId(index);
                    edge->setVertex(0, vertices.at(frame->keyframe_id_));    // pose
                    edge->setVertex(1, vertices_landmarks.at(landmark_id));  // landmark
                    edge->setMeasurement(toVec2(feat->position_.pt));
                    edge->setInformation(Mat22::Identity());
                    auto rk = new g2o::RobustKernelHuber();
                    rk->setDelta(chi2_th);
                    edge->setRobustKernel(rk);
                    edges_and_features.insert({edge, feat});
                    optimizer.addEdge(edge);
                    index++;
                }
            else delete edge;
                
        }
    }

    // 6. 执行优化并剔除 outlier
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    int cnt_outlier = 0, cnt_inlier = 0;
    int iteration = 0;
    while (iteration < 5) {
        cnt_outlier = 0;
        cnt_inlier = 0;
        // determine if we want to adjust the outlier threshold
        for (auto &ef : edges_and_features) {
            if (ef.first->chi2() > chi2_th) {
                cnt_outlier++;
            } else {
                cnt_inlier++;
            }
        }
        double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
        if (inlier_ratio > 0.5) {
            break;
        } else {
            chi2_th *= 2;
            iteration++;
        }
    }

    for (auto &ef : edges_and_features) {
        if (ef.first->chi2() > chi2_th) {
            ef.second->is_outlier_ = true;
            // remove the observation
            ef.second->map_point_.lock()->RemoveObservation(ef.second);
        } else {
            ef.second->is_outlier_ = false;
        }
    }

    LOG(INFO) << "Outlier/Inlier in optimization: " << cnt_outlier << "/"
              << cnt_inlier;



    // 7. 更新位姿和路标点lanrmark位置
    for (auto &v : vertices) {
        keyframes.at(v.first)->SetPose(v.second->estimate());
    }
    for (auto &v : vertices_landmarks) {
        landmarks.at(v.first)->SetPos(v.second->estimate());
    }
}

}  // namespace myslam
