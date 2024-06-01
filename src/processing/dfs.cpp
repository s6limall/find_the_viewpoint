//
// Created by ayush on 5/21/24.
//

#include <spdlog/spdlog.h>
#include <Eigen/Core>




Eigen::Vector3d calculate_new_center(const Eigen::Vector3d & A, const Eigen::Vector3d & B, const Eigen::Vector3d & C) {
    Eigen::Vector3d AB = B - A;
    Eigen::Vector3d AC = C - A;

    Eigen::Vector3d normal = AB; //.cross(AC);
    
    normal.normalize();
    
    Eigen::Vector3d new_center = normal; // This assumes radius is 1 and the original center is at (0,0,0)
    
    return new_center;
}


const Eigen::Vector3d depth_first_search(const Eigen::Vector3d & p1, const Eigen::Vector3d & p2, const Eigen::Vector3d & p3, double best_score){
    //Check Triangle score if better dont return

    //CHeck triangles in triangle
    Eigen::Vector3d p4 = calculate_new_center(p1,p2,p3);

    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> edges = {
        {p1, p2},
        {p2, p3},
        {p3, p1}
    };

    Eigen::Vector3d best_candiate;
    for (const auto &edge : edges) {
        // Call new function if better add score if score well 
        best_candiate = depth_first_search(edge.first, edge.second, p4, best_score);
        //compare score if better store candidate
    }
    return best_candiate;
}