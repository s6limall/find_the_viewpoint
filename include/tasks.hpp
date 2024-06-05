//
// Created by ayush on 5/21/24.
//

#ifndef TASKS_HPP
#define TASKS_HPP

#include <string>

class Tasks {
public:
    void execute(const std::string& taskName, const std::string& object_name, int test_num);

private:
    void task1(const std::string& object_name, int test_num);
};

#endif // TASKS_HPP
