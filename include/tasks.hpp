//
// Created by ayush on 5/21/24.
//

#ifndef TASKS_HPP
#define TASKS_HPP

#include <string>

// The Tasks class is responsible for managing and performing viewpoint evaluations for 3D objects.
class Tasks {
public:
    // Performs viewpoint evaluation based on a set of parameters.
    void performViewpointEvaluation(const std::string &task_name, const std::string &object_name, int test_num,
                                    const std::string &view_file_path);
};

#endif // TASKS_HPP
