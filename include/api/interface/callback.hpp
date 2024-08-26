// File: api/interface/callback.hpp

#ifndef CALLBACK_HPP
#define CALLBACK_HPP

#include <functional>

#include "types/viewpoint.hpp"

class Callback {
public:
    using CallbackFunction = std::function<void(const ViewPoint<> &)>;
    virtual ~Callback() = default;

    virtual void registerCallback(CallbackFunction callback) = 0;
    virtual void invoke(const ViewPoint<>& viewpoint) const = 0;

};

#endif //CALLBACK_HPP
