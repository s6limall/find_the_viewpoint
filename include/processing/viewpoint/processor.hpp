// File: processing/viewpoint/processor.hpp

#ifndef VIEWPOINT_PROCESSOR_HPP
#define VIEWPOINT_PROCESSOR_HPP

namespace processing::viewpoint {

    class Processor {
    public:
        Processor(const Processor &) = delete;

        Processor &operator=(const Processor &) = delete;

        static void process();

    private:
        Processor() = default;

        static void initialize();
    };
}

#endif //VIEWPOINT_PROCESSOR_HPP
