// File: api/interface/publisher.hpp

#ifndef PUBLISHER_HPP
#define PUBLISHER_HPP
#include "types/viewpoint.hpp"

class Publisher {
public:
    virtual ~Publisher() = default;

    virtual void publish(const ViewPoint<>& viewpoint) const = 0;
};

#endif //PUBLISHER_HPP
